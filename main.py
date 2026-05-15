import asyncio
from typing import Dict, List, Any
from langgraph.graph import StateGraph, START, END
import json
import re
import uuid

from agents import run_planner_async, run_analyst_async, run_reviewer_async, run_editor_async, run_researcher_async
from ragas_evaluator import RagasEvaluationError, evaluate_faithfulness
from tools import MilvusRAGHelper
from models import ResearchState, SectionState
from logger import (
    append_planner_log, append_researcher_log, append_analyst_log,
    append_reviewer_log, append_editor_log, init_log_file, _ensure_log_folder
)

def _safe_collection_name(prefix: str, title: str, max_len: int = 60) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", title).strip("_")
    if not safe:
        safe = "section"
    safe = f"{prefix}_{safe}"
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_")
    return safe

def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(0, end - max(0, overlap))
    return chunks

def _build_rag_docs_from_results(results: List[Dict[str, Any]]) -> List[str]:
    """
    将搜索结果构造成可检索的文档块。

    注意：将 [数据源X] 前缀写入文档块中，便于 Analyst 直接引用来源编号。
    """
    docs: List[str] = []
    for i, res in enumerate(results, start=1):
        title = (res.get("title") or "").strip()
        url = (res.get("url") or "").strip()
        content = (res.get("content") or "").strip()
        base = f"[数据源{i}] 标题: {title}\n来源链接: {url}\n内容:\n{content}".strip()
        for chunk in _chunk_text(base):
            docs.append(chunk)
    return docs

def _extract_query_keywords(query: str) -> List[str]:
    query = (query or "").strip().lower()
    if not query:
        return []

    # 英文/数字关键词（过滤掉太短的噪声）
    en = re.findall(r"[a-z0-9]{3,}", query)
    # 中文关键词：连续 2+ 字，避免单字噪声
    zh = re.findall(r"[\u4e00-\u9fff]{2,}", query)

    # 去重保持顺序
    keywords: List[str] = []
    for k in zh + en:
        if k and k not in keywords:
            keywords.append(k)
    return keywords

def _lexical_rag_retrieve(docs: List[str], query: str, top_k: int = 6) -> List[str]:
    """
    无 embedding 环境下的降级版 RAG：用简单词频匹配做召回。
    目的：保证“资料变多时仍能压缩输入”，且不依赖外部模型下载。
    """
    if not docs:
        return []

    keywords = _extract_query_keywords(query)
    if not keywords:
        return docs[:top_k]

    scored = []
    for d in docs:
        dl = d.lower()
        score = 0
        for k in keywords:
            score += dl.count(k.lower())
        scored.append((score, d))

    scored.sort(key=lambda x: x[0], reverse=True)
    # 如果全是 0 分，直接取前 top_k
    if scored and scored[0][0] == 0:
        return docs[:top_k]
    return [d for _, d in scored[:top_k]]

# 真实的 Planner 节点：动态生成提纲
async def planner_node(state: ResearchState) -> ResearchState:
    print("\n--> 节点执行: planner")
    topic = state.topic

    if not state.sections:
        print(f"    [Planner] 正在为主题 '{topic}' 动态生成研究大纲...")
        outline = await run_planner_async(topic)
        print(f"    [Planner] 成功生成大纲: {outline}")

        for sec_title in outline:
            state.sections[sec_title] = SectionState(title=sec_title)

        try:
            append_planner_log(topic, topic, outline)
        except Exception as e:
            print(f"    [Planner] 日志记录失败: {e}")

    return state

async def researcher_node(state: ResearchState) -> ResearchState:
    print("\n--> 节点执行: researcher (ReAct Multi-Hop)")
    topic = state.topic

    sections_to_search = [title for title, sec in state.sections.items() if not sec.research_data]

    if not sections_to_search:
        print("    [Researcher] 所有章节均已有基础研究数据。")
        return state

    print(f"    [Researcher] ReAct 多跳搜索启动，目标章节: {sections_to_search}")

    async def search_section(sec_title: str):
        query = f"{topic} {sec_title}"
        try:
            raw_results, formatted_data = await run_researcher_async(topic, sec_title)
            return sec_title, raw_results, formatted_data
        except Exception as e:
            print(f"    [Researcher] 章节 '{sec_title}' ReAct 检索失败: {e}")
            return sec_title, [], f"关于【{sec_title}】的联网检索失败，请依赖大模型的内部知识进行分析。"

    results = await asyncio.gather(*(search_section(sec) for sec in sections_to_search))

    raw_results_map: Dict[str, List[Dict[str, Any]]] = {}
    for sec_title, raw_results, formatted_data in results:
        raw_results_map[sec_title] = raw_results
        state.sections[sec_title].research_data = formatted_data

    try:
        rag_db = f"./milvus_local_{uuid.uuid4().hex}.db"
        rag = MilvusRAGHelper(db_path=rag_db)

        for idx, sec_title in enumerate(sections_to_search, start=1):
            sec = state.sections[sec_title]
            sec_results = raw_results_map.get(sec_title, [])
            if not sec_results:
                sec.rag_context = ""
                continue

            collection_name = _safe_collection_name(prefix=f"sec{idx}", title=sec_title)
            docs = _build_rag_docs_from_results(sec_results)
            if not docs:
                sec.rag_context = ""
                continue

            rag.init_collection(collection_name)
            rag.add_documents(collection_name, docs)

            top_k = min(6, len(docs))
            retrieved = rag.search(collection_name, query=f"{topic} {sec_title}", top_k=top_k)
            sec.rag_context = "\n\n---\n\n".join(retrieved)
    except Exception as e:
        print(f"    [Researcher] 向量 RAG 召回失败，降级为 lexical RAG（原因: {e}）")
        for sec_title in sections_to_search:
            sec = state.sections[sec_title]
            sec_results = raw_results_map.get(sec_title, [])
            docs = _build_rag_docs_from_results(sec_results)
            top_k = min(6, len(docs)) if docs else 0
            retrieved = _lexical_rag_retrieve(docs, query=f"{topic} {sec_title}", top_k=top_k) if top_k else []
            sec.rag_context = "\n\n---\n\n".join(retrieved)

    try:
        for sec_title, raw_results, formatted_data in results:
            append_researcher_log(topic, topic, sec_title, raw_results, formatted_data)
    except Exception as e:
        print(f"    [Researcher] 日志记录失败: {e}")

    return state

# Analyst 节点：并行处理所有未通过的章节
async def analyst_node(state: ResearchState) -> ResearchState:
    print("\n--> 节点执行: analyst (Async Parallel)")
    topic = state.topic
    
    # 找出需要处理的章节（没有草稿，或者存在 critique 需要重写的）
    sections_to_process = [
        title for title, sec in state.sections.items() 
        if not sec.draft or (sec.critique and not sec.is_approved)
    ]
            
    if not sections_to_process:
        print("    [Analyst] 所有章节已完成，无需处理。")
        return state
        
    print(f"    [Analyst] 正在并发撰写/重写章节: {sections_to_process}")
    
    async def process_section(sec_title: str):
        sec = state.sections[sec_title]
        research_payload = sec.research_data
        if getattr(sec, "rag_context", ""):
            research_payload = (
                "【RAG 召回关键片段（优先参考，用于压缩输入与提升相关性）】\n"
                f"{sec.rag_context}\n\n"
                "【原始资料（用于追溯全部数据源与链接）】\n"
                f"{sec.research_data}"
            )
        draft = await run_analyst_async(topic, sec_title, research_payload, sec.critique)
        return sec_title, draft
        
    results = await asyncio.gather(*(process_section(sec) for sec in sections_to_process))
    
    for sec_title, draft in results:
        sec = state.sections[sec_title]
        sec.draft = draft
        sec.critique = ""
        sec.is_approved = False

    try:
        for sec_title, draft in results:
            sec = state.sections[sec_title]
            append_analyst_log(topic, topic, sec_title, sec.research_data, draft, sec.critique)
    except Exception as e:
        print(f"    [Analyst] 日志记录失败: {e}")

    return state

# Reviewer 节点：并行审查所有新草稿
async def reviewer_node(state: ResearchState) -> ResearchState:
    print("\n--> 节点执行: reviewer (Async Parallel)")
    topic = state.topic
    
    # 找出有草稿且还没有被 review（或处于未 approve 且没 critique 状态）的章节
    sections_to_review = [
        title for title, sec in state.sections.items() 
        if sec.draft and not sec.is_approved and not sec.critique
    ]
    
    if not sections_to_review:
         print("    [Reviewer] 没有需要审查的新草稿。")
         return state
         
    print(f"    [Reviewer] 正在并发审查章节: {sections_to_review}")
    
    async def review_section(sec_title: str):
        sec = state.sections[sec_title]
        result = await run_reviewer_async(topic, sec_title, sec.draft, sec.research_data)
        return sec_title, result
        
    results = await asyncio.gather(*(review_section(sec) for sec in sections_to_review))
    
    all_approved = True
    for sec_title, res in results:
        is_approved = res.get("is_approved", False)
        feedback = res.get("feedback", "")
        
        sec = state.sections[sec_title]
        print(f"    [Reviewer] 章节 '{sec_title}' 结果: 通过={is_approved}")
        
        sec.is_approved = is_approved
        if not is_approved:
            all_approved = False
            sec.critique = feedback
            print(f"               增强指令: {feedback}")
            
    if not all_approved:
        state.revision_count += 1
        print(f"    [Reviewer] 存在未通过的章节，当前总迭代次数: {state.revision_count}")

    try:
        for sec_title, res in results:
            sec = state.sections[sec_title]
            append_reviewer_log(topic, topic, sec_title, sec.draft, res.get("is_approved", False), res.get("feedback", ""), sec.research_data)
    except Exception as e:
        print(f"    [Reviewer] 日志记录失败: {e}")

    return state


def _normalize_citations_in_drafts(drafts: Dict[str, str], sections: Dict[str, Any]) -> Dict[str, str]:
    _, local_to_global = _parse_all_sources(sections)
    normalized = {}
    for sec_title, draft in drafts.items():
        text = draft
        for mapping_key, global_id in local_to_global.items():
            map_sec, map_local = mapping_key.split('|||', 1)
            if map_sec != sec_title:
                continue
            text = re.sub(r'\[' + re.escape(map_local) + r'\]', f'[{global_id}]', text)
        normalized[sec_title] = text
    return normalized



def _source_identity(title: str, url: str) -> str:
    normalized_url = (url or "").strip().rstrip(".,;，。；)）]】").lower()
    if normalized_url:
        return f"url:{normalized_url}"
    normalized_title = re.sub(r"\s+", " ", title or "").strip().lower()
    return f"title:{normalized_title}"


def _extract_cited_ids(report: str) -> set:
    return set(re.findall(r"(?<!\!)\[(\d+)\]", report or ""))


def _sort_numeric_strings(values: set) -> List[str]:
    return sorted(values, key=lambda x: int(x) if x.isdigit() else x)


def _validate_citation_consistency(report: str, source_list: List[Dict[str, str]]) -> Dict[str, List[str]]:
    cited_ids = _extract_cited_ids(report)
    source_ids = {src["id"] for src in source_list}
    return {
        "missing_references": _sort_numeric_strings(cited_ids - source_ids),
        "unused_references": _sort_numeric_strings(source_ids - cited_ids),
    }


def _parse_all_sources(sections: Dict[str, Any]) -> tuple:
    source_list: List[Dict[str, str]] = []
    local_to_global: Dict[str, str] = {}
    source_identity_to_global: Dict[str, str] = {}
    global_counter = 0

    for sec_title, sec in sections.items():
        rd = getattr(sec, "research_data", "") or ""
        blocks = re.split(r"\n(?=\[[^\]\n]*\d+\]\s*)", rd)
        for block in blocks:
            match = re.match(r"\[([^\]\n]*\d+)\]\s*(?:[^:\n：]{0,30})[:：]\s*(.+?)(?:\n|$)", block)
            if not match:
                continue

            local_id = match.group(1).strip()
            source_title = match.group(2).strip()
            url_match = re.search(r"https?://\S+", block)
            source_url = url_match.group(0).strip().rstrip(".,;，。；)）]】") if url_match else ""
            identity = _source_identity(source_title, source_url)

            if identity in source_identity_to_global:
                global_id = source_identity_to_global[identity]
                for src in source_list:
                    if src["id"] == global_id:
                        sections_used = src["section"].split(" / ")
                        if sec_title not in sections_used:
                            src["section"] = f'{src["section"]} / {sec_title}'
                        break
            else:
                global_counter += 1
                global_id = str(global_counter)
                source_identity_to_global[identity] = global_id
                source_list.append({
                    "id": global_id,
                    "title": source_title,
                    "url": source_url,
                    "section": sec_title,
                })

            mapping_key = f"{sec_title}|||{local_id}"
            local_to_global[mapping_key] = global_id

    return source_list, local_to_global


def _replace_references_section(report: str, sections: Dict[str, Any]) -> str:
    source_list, _ = _parse_all_sources(sections)
    if not source_list:
        return report

    cited_ids = _extract_cited_ids(report)
    if cited_ids:
        source_list = [src for src in source_list if src["id"] in cited_ids]

    if not source_list:
        return report

    lines: List[str] = []
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 参考资料")
    lines.append("")

    for src in source_list:
        sid = src["id"]
        title = src["title"]
        url = src["url"]
        section = src["section"]
        if url:
            lines.append(f"- **[{sid}]** [{title}]({url})")
        else:
            lines.append(f"- **[{sid}]** {title}")
        lines.append(f"  *章节：{section}*")
        lines.append("")
    lines.append("")

    new_ref_block = "\n".join(lines)

    ref_pattern = r"\n*#{1,3}\s*(?:参考资料|å‚è€ƒèµ„æ–™)\s*\n.*?(?=\n#{1,3}\s|\Z)"
    ref_match = re.search(ref_pattern, report, re.DOTALL | re.IGNORECASE)
    if ref_match:
        report = report[:ref_match.start()].rstrip() + "\n" + new_ref_block + "\n" + report[ref_match.end():].lstrip()
    else:
        report = report.rstrip() + "\n" + new_ref_block + "\n"

    return report


def _build_evaluation_contexts(sections: Dict[str, Any], max_contexts: int = 20, max_chars: int = 1200) -> List[str]:
    contexts: List[str] = []
    for sec_title, sec in sections.items():
        preferred_context = getattr(sec, "rag_context", "") or getattr(sec, "research_data", "")
        for chunk in _chunk_text(preferred_context, chunk_size=max_chars, overlap=0):
            cleaned = chunk.strip()
            if cleaned:
                contexts.append(f"章节：{sec_title}\n{cleaned}")
            if len(contexts) >= max_contexts:
                return contexts
    return contexts


# Editor 节点
async def editor_node(state: ResearchState) -> ResearchState:
    print("\n--> 节点执行: editor")
    topic = state.topic

    drafts = {title: sec.draft for title, sec in state.sections.items() if sec.draft}

    drafts = _normalize_citations_in_drafts(drafts, state.sections)

    conflict_resolutions = state.conflict_resolutions

    print(f"    [Editor] 正在整合 {len(drafts)} 个章节并生成最终报告...")
    final_report = await run_editor_async(topic, drafts, conflict_resolutions)

    source_list, _ = _parse_all_sources(state.sections)
    citation_check = _validate_citation_consistency(final_report, source_list)
    if citation_check["missing_references"]:
        print(f"    [Editor] 引用校验警告：正文引用了不存在的来源编号 {citation_check['missing_references']}")
    if citation_check["unused_references"]:
        print(f"    [Editor] 引用校验提示：以下来源未被正文实际引用 {citation_check['unused_references']}")

    final_report = _replace_references_section(final_report, state.sections)

    state.final_report = final_report

    try:
        append_editor_log(topic, topic, final_report, len(drafts))
    except Exception as e:
        print(f"    [Editor] 日志记录失败: {e}")

    print("    [Editor] 最终报告整合完毕，等待外部导出。")
    return state


async def ragas_evaluator_node(state: ResearchState) -> ResearchState:
    print("\n--> 节点执行: ragas_evaluator")
    contexts = _build_evaluation_contexts(state.sections)
    try:
        state.faithfulness_score = await evaluate_faithfulness(
            user_input=state.topic,
            response=state.final_report,
            retrieved_contexts=contexts,
        )
        state.faithfulness_error = ""
        print(f"    [RAGAS] Faithfulness 评估完成: {state.faithfulness_score:.4f}")
    except RagasEvaluationError as e:
        state.faithfulness_score = None
        state.faithfulness_error = str(e)
        print(f"    [RAGAS] Faithfulness 评估失败: {e}")
    except Exception as e:
        state.faithfulness_score = None
        state.faithfulness_error = f"RAGAS 评估运行异常: {e}"
        print(f"    [RAGAS] Faithfulness 评估异常: {e}")
    return state

# 条件路由逻辑
def should_revise(state: ResearchState) -> str:
    # 检查是否有未通过且带有 critique 的章节
    has_critique = any(sec.critique for sec in state.sections.values() if not sec.is_approved)
    revision_count = state.revision_count
    
    if has_critique:
        if revision_count < 3:
            print(f"    [条件路由] 存在未通过章节且 revision_count ({revision_count}) < 3，退回 analyst 进行修正")
            return "analyst"
        else:
            print(f"    [条件路由] revision_count ({revision_count}) >= 3！触发 Editor 强制介入")
            # 记录冲突供 Editor 解决
            conflicts = []
            for title, sec in state.sections.items():
                if not sec.is_approved and sec.critique:
                    conflicts.append(f"章节【{title}】的最后冲突记录：{sec.critique}")
                    # 清空 critique 避免死循环
                    sec.critique = ""
            state.conflict_resolutions = "\n".join(conflicts)
            return "editor"
            
    print(f"    [条件路由] 所有章节审查通过，前往 editor")
    return "editor"


def should_run_ragas_evaluation(state: ResearchState) -> str:
    if state.enable_ragas_evaluation:
        print("    [条件路由] 已开启 RAGAS 评估，前往 ragas_evaluator")
        return "ragas_evaluator"
    print("    [条件路由] 未开启 RAGAS 评估，结束流程")
    return "end"

# ==========================================
# 组装图并运行
# ==========================================
workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("editor", editor_node)
workflow.add_node("ragas_evaluator", ragas_evaluator_node)

workflow.add_edge(START, "planner")
workflow.add_edge("planner", "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "reviewer")

workflow.add_conditional_edges(
    "reviewer",
    should_revise,
    {
        "analyst": "analyst",
        "editor": "editor"
    }
)

workflow.add_conditional_edges(
    "editor",
    should_run_ragas_evaluation,
    {
        "ragas_evaluator": "ragas_evaluator",
        "end": END
    }
)
workflow.add_edge("ragas_evaluator", END)

# 编译图结构
app = workflow.compile()
