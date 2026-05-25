import os
import json
import asyncio
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

from tools import tool_manager, web_search

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_BASE,
    temperature=0.3
)

# ==========================================
# 0. Planner Agent (异步) - 动态生成提纲
# ==========================================
async def run_planner_async(topic: str) -> List[str]:
    """
    Planner Agent: 根据用户输入的主题，动态生成一份合理的研究报告大纲。
    返回一个包含章节标题的字符串列表。
    """
    system_prompt = """你是一个资深的研究规划师（Planner Agent）。
你的任务是根据用户给定的研究主题，拆解出一份逻辑清晰、结构合理的深度研究报告大纲。
要求：
1. 只需要输出大纲的章节标题，不要输出具体的正文内容。
2. 章节数量控制在 3 到 5 个之间，不要过于冗长。
3. 你必须输出一个纯 JSON 对象，不要包含 Markdown 代码块标记（如 ```json），格式如下：
{
    "outline": ["引言与背景", "核心技术分析", "市场现状与竞品", "未来发展趋势"]
}"""

    user_prompt = f"研究主题: {topic}\n请生成研究大纲："
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = await llm.with_kwargs(response_format={"type": "json_object"}).ainvoke(messages)
        content = response.content.strip()
    except Exception:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        result = json.loads(content)
        return result.get("outline", [f"{topic} 概述", "深入分析", "总结与展望"])
    except json.JSONDecodeError:
        return [f"{topic} 概述", "深入分析", "总结与展望"]


# ==========================================
# 1. Analyst Agent (异步)
# ==========================================
async def run_analyst_async(topic: str, section_title: str, research_data: str, instruction: str = "") -> str:
    """
    Analyst Agent: 接收 research_data 和子章节标题，生成深度分析。
    instruction: 来自 Reviewer 的反馈指令（如果这是重写轮次）
    """
    system_prompt = """你是一个顶级的行业分析师（Analyst Agent）。
你的任务是基于提供的原始研究数据（research_data），针对特定子章节生成深度分析。
要求：
1. 使用专业术语，逻辑严密。
2. 每条重要结论必须明确引用搜索到的原始数据（例如：[数据源1]）。
3. 不要捏造数据，如果提供的数据不足以得出结论，请如实说明。"""

    user_prompt = f"""
主题: {topic}
当前正在撰写的章节: {section_title}

[原始研究数据]
{research_data}

"""
    if instruction:
        user_prompt += f"\n[Reviewer 的修改指令]\n{instruction}\n请严格按照此指令修改并重写本章节。\n"

    user_prompt += "\n请根据以上信息，撰写该章节的深度分析报告："
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    return response.content


# ==========================================
# 2. Reviewer Agent (异步)
# ==========================================
async def run_reviewer_async(topic: str, section_title: str, draft_content: str, research_data: str) -> Dict[str, Any]:
    """
    Reviewer Agent: 扮演苛刻的资深分析师，检查幻觉、数据支撑、逻辑矛盾。
    返回结构化的 JSON 包含 is_approved 和 feedback。
    """
    system_prompt = """你是一个极度苛刻的资深分析师（Reviewer Agent）。
你的职责是严格审查 Analyst 生成的草稿。你需要检查：
1. 是否存在“幻觉”（即草稿中出现了原始研究数据中没有的内容）。
2. 结论是否有充分的数据支撑。
3. 逻辑是否前后矛盾。

你必须输出一个纯 JSON 对象，不要包含 Markdown 代码块标记（如 ```json），必须严格遵守以下格式：
{
    "is_approved": true或false,
    "feedback": "如果不通过，将其转化为给 Analyst 的具体重写指令(Instruction)，要求其检索或修改特定信息；如果通过，简短评价"
}"""

    user_prompt = f"""
主题: {topic}
章节: {section_title}

[原始研究数据]
{research_data}

[Analyst 生成的草稿]
{draft_content}

请严格审查上述草稿，并按照要求输出 JSON:
"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    try:
        response = await llm.with_kwargs(response_format={"type": "json_object"}).ainvoke(messages)
        content = response.content.strip()
    except Exception:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    try:
        result = json.loads(content)
        if "is_approved" not in result or "feedback" not in result:
            return {"is_approved": False, "feedback": "Reviewer 输出的 JSON 缺少必要字段。"}
        return result
    except json.JSONDecodeError:
        return {
            "is_approved": False,
            "feedback": f"Reviewer 未能输出合法的 JSON 格式。原始内容: {content}"
        }


# ==========================================
# 3. Editor Agent (异步)
# ==========================================
async def run_editor_async(topic: str, drafts: Dict[str, str], conflict_resolutions: str = "") -> str:
    """
    Editor Agent: 负责将所有通过审核的章节整合成 Markdown 长报告。
    如果在多次迭代后仍有未解决的冲突，Editor 会进行人工化处理。
    """
    system_prompt = """你是一个资深的报告编辑（Editor Agent）。
你的任务是将多篇章节草稿整合为一篇完整的 Markdown 格式长报告。
要求：
1. 报告必须包含自动生成的目录（Table of Contents, TOC），使用 Markdown 锚点链接。
2. 内容排版要美观，层级清晰，段落之间用空行分隔，确保 Markdown 渲染后换行正常。
3. 保留草稿中已有的引用标记（如 [1]、[2] 等），不要修改或删除它们。
4. 润色段落间的过渡，使整篇文章读起来流畅自然。
5. 不要在报告末尾添加\"参考资料\"章节，这将在后续由系统自动生成。"""

    drafts_text = ""
    for section_title, content in drafts.items():
        drafts_text += f"\n\n### 章节：{section_title}\n{content}"

    user_prompt = f"""
报告主题: {topic}

以下是各个章节的定稿内容：
{drafts_text}
"""
    if conflict_resolutions:
        system_prompt += "\n5. 注意：某些章节存在未解决的分析冲突。作为 Editor，请你在整合时平滑处理这些冲突，客观陈述事实，避免绝对化结论。"
        user_prompt += f"\n[未解决的冲突记录]\n{conflict_resolutions}\n"

    user_prompt += "\n请整合成最终的 Markdown 长报告："
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    return response.content

REACT_SYSTEM_PROMPT = """你是一个深度研究搜索智能体（Researcher Agent），遵循 ReAct（Reasoning + Acting）范式。
你的任务是通过多轮迭代搜索，为特定研究章节收集全面、深入的网络资料。

工作流程：
1. 分析当前已收集的资料，识别信息缺口
2. 制定新的搜索查询，从不同角度深入挖掘
3. 当你认为资料足够充分时（通常 2-4 轮搜索后），输出结束指令

要求：
- 每轮只执行一次搜索，查询应具体且与前几轮不重复
- 优先从不同维度切入：定义背景 → 技术细节 → 案例数据 → 观点争议 → 最新进展
- 查询语言请使用中文（如果主题是中文语境）或英文

输出格式（纯 JSON，不要 Markdown 代码块）：
- 继续搜索：{"action": "search", "query": "你的搜索查询词"}
- 结束搜索：{"action": "finish", "summary": "简要总结已收集资料的关键发现"}"""


async def _run_researcher_async_legacy(topic: str, section_title: str, max_hops: int = 4) -> Tuple[List[Dict[str, Any]], str]:
    """
    ReAct Agent: 多跳搜索研究器
    通过 Reasoning + Acting 循环进行多轮深层搜索，逐步深入主题的各维度。
    max_hops: 最大搜索轮数（默认 4，即最多 4 次搜索）
    返回: (原始搜索结果列表, 格式化后的研究数据字符串)
    """
    all_results: List[Dict[str, Any]] = []
    conversation: List[Any] = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),
        HumanMessage(content=f"研究主题: {topic}\n当前章节: {section_title}\n请开始第一轮搜索，从最核心的概念或背景信息入手。")
    ]

    for hop in range(max_hops):
        print(f"    [ReAct] 第 {hop + 1}/{max_hops} 轮推理...")

        try:
            response = await llm.ainvoke(conversation)
        except Exception as e:
            print(f"    [ReAct] LLM 调用失败: {e}")
            break

        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            decision = json.loads(content)
        except json.JSONDecodeError:
            print(f"    [ReAct] JSON 解析失败，原始输出: {content[:200]}...")
            continue

        action = decision.get("action", "")

        if action == "finish":
            summary = decision.get("summary", "")
            print(f"    [ReAct] Agent 决定结束搜索。总结: {summary[:100]}...")
            break

        elif action == "search":
            query = decision.get("query", "")
            if not query:
                print(f"    [ReAct] 搜索动作为空 query，跳过")
                continue

            print(f"    [ReAct] 执行搜索: {query}")
            try:
                results = await asyncio.to_thread(web_search, query, max_results=5)
            except Exception as e:
                print(f"    [ReAct] 搜索失败: {e}")
                observation = f"搜索 '{query}' 执行失败: {str(e)}"
                conversation.append(AIMessage(content=content))
                conversation.append(HumanMessage(content=f"[观察结果]\n{observation}"))
                continue

            all_results.extend(results)

            observation_parts = [f"搜索结果（共 {len(results)} 条）:"]
            for i, res in enumerate(results):
                title = res.get('title', '无标题')
                snippet = res.get('content', '无摘要')[:300]
                url = res.get('url', '')
                observation_parts.append(f"[结果{i+1}] {title}\n  {snippet}\n  链接: {url}")
            observation = "\n".join(observation_parts)

            conversation.append(AIMessage(content=content))
            conversation.append(HumanMessage(content=f"[观察结果]\n{observation}\n\n请分析以上结果，决定下一步：是继续搜索新角度，还是资料已充足可以结束？"))
        else:
            print(f"    [ReAct] 未知动作: {action}")
            break

    source_index = 1
    formatted_parts = [f"以下是关于【{section_title}】的多跳深度搜索结果（共 {len(all_results)} 条原始资料）：\n"]
    seen_urls = set()
    for res in all_results:
        url = res.get('url', '')
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        title = res.get('title', '无标题')
        content = res.get('content', '无摘要内容')
        formatted_parts.append(
            f"[数据源{source_index}] 标题: {title}\n"
            f"内容: {content}\n"
            f"来源链接: {url}\n"
        )
        source_index += 1

    formatted_data = "\n".join(formatted_parts)
    print(f"    [ReAct] 多跳搜索完成，共执行 {min(hop + 1, max_hops)} 轮，收集 {source_index - 1} 条去重资料。")
    return all_results, formatted_data

async def run_researcher_async(topic: str, section_title: str, max_hops: int = 4) -> Tuple[List[Dict[str, Any]], str]:
    """
    Framework-driven ReAct researcher.

    This overrides the legacy hand-written ReAct loop above while preserving the
    same public contract used by main.py:
    (raw search results, formatted research data).
    """
    all_results: List[Dict[str, Any]] = []
    search_count = 0

    @tool
    def researcher_web_search(query: str, max_results: int = 5) -> str:
        """Search web information for the current deep-research section."""
        nonlocal search_count
        if search_count >= max_hops:
            return json.dumps(
                {
                    "status": "limit_reached",
                    "message": f"Search limit reached: {max_hops} calls.",
                },
                ensure_ascii=False,
            )

        search_count += 1
        print(f"    [ReAct] framework tool search {search_count}/{max_hops}: {query}")
        try:
            results = web_search(query, max_results=max_results)
        except Exception as e:
            print(f"    [ReAct] search failed: {e}")
            return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)

        all_results.extend(results)
        return json.dumps({"status": "success", "results": results}, ensure_ascii=False)

    framework_prompt = (
        "You are a deep-research search agent following the ReAct pattern. "
        "Your task is to collect comprehensive web material for one report section. "
        "Use the researcher_web_search tool when more evidence is needed. "
        "Do not output JSON action commands. The framework will execute tool calls for you. "
        "Search from different angles: background, technical details, cases/data, debates, and recent progress. "
        "Each search query should be specific and should avoid repeating previous searches. "
        "When enough material has been collected, stop using tools and summarize the key findings."
    )

    agent = create_react_agent(
        model=llm,
        tools=[researcher_web_search],
        prompt=framework_prompt,
    )

    user_prompt = (
        f"Research topic: {topic}\n"
        f"Current section: {section_title}\n"
        f"Use researcher_web_search for up to {max_hops} focused searches. "
        "Start from core concepts or background, then explore different angles. "
        "When the collected material is sufficient, stop calling tools and summarize the key findings."
    )

    print(f"    [ReAct] framework agent started, max search calls: {max_hops}")
    try:
        await agent.ainvoke(
            {"messages": [HumanMessage(content=user_prompt)]},
            config={"recursion_limit": max_hops * 4 + 6},
        )
    except Exception as e:
        print(f"    [ReAct] framework agent failed: {e}")

    source_index = 1
    formatted_parts = [
        f"以下是关于【{section_title}】的多跳深度搜索结果（共 {len(all_results)} 条原始资料）：\n"
    ]
    seen_urls = set()
    for res in all_results:
        url = res.get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)

        title = res.get("title", "无标题")
        content = res.get("content", "无摘要内容")
        formatted_parts.append(
            f"[数据源{source_index}] 标题: {title}\n"
            f"内容: {content}\n"
            f"来源链接: {url}\n"
        )
        source_index += 1

    formatted_data = "\n".join(formatted_parts)
    print(
        f"    [ReAct] framework search complete, tool calls: {search_count}, "
        f"unique sources: {source_index - 1}."
    )
    return all_results, formatted_data


if __name__ == "__main__":
    print("Agent 模块已加载，可被 main.py 调用。")
