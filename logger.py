import os
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

LOG_FOLDER = "Log"

def _ensure_log_folder():
    Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)

def _sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for c in invalid_chars:
        name = name.replace(c, '_')
    return name.strip()

def _get_log_path(keyword: str) -> str:
    _ensure_log_folder()
    safe_keyword = _sanitize_filename(keyword)
    return os.path.join(LOG_FOLDER, f"{safe_keyword}.md")

def _format_content_for_markdown(content: str) -> str:
    content = content.replace('```', '~~~')
    return content

def init_log_file(keyword: str, topic: str) -> str:
    log_path = _get_log_path(keyword)
    file_exists = os.path.exists(log_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not file_exists:
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"# 研究日志：{topic}\n\n")
            f.write(f"> 创建时间：{timestamp}\n\n")
            f.write("---\n\n")
    else:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n---\n\n")
            f.write(f"## 续写时间：{timestamp}\n\n")

    return log_path

def append_planner_log(keyword: str, topic: str, outline: List[str], sources: List[Dict[str, Any]] = None) -> str:
    log_path = init_log_file(keyword, topic)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = [
        f"### Planner 节点执行 | {timestamp}\n",
        f"**主题：** {topic}\n",
        f"**操作：** 动态生成研究报告大纲\n\n",
        "#### 1. 资料来源\n"
    ]

    if sources:
        for i, src in enumerate(sources, 1):
            content.append(f"- **来源{i}：** {src.get('title', 'N/A')}\n")
            content.append(f"  - 链接：{src.get('url', 'N/A')}\n")
            content.append(f"  - 摘要：{src.get('content', 'N/A')[:200]}...\n" if len(src.get('content', '')) > 200 else f"  - 摘要：{src.get('content', 'N/A')}\n")
    else:
        content.append("- 无外部资料来源（基于大模型内部知识生成）\n")

    content.append("\n#### 2. 识别的问题\n")
    content.append(f"- 根据主题 **\"{topic}\"** 拆解研究框架\n")

    content.append("\n#### 3. 生成的解决方案/输出\n")
    content.append("**生成的大纲结构：**\n")
    for i, section in enumerate(outline, 1):
        content.append(f"{i}. {section}\n")

    content.append("\n---\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(content)

    return log_path

def append_researcher_log(keyword: str, topic: str, section_title: str, search_results: List[Dict[str, Any]], formatted_data: str) -> str:
    log_path = init_log_file(keyword, topic)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = [
        f"### Researcher 节点执行 | {timestamp}\n",
        f"**主题：** {topic}\n",
        f"**章节：** {section_title}\n",
        f"**操作：** 并发联网检索与资料整理\n\n",
        "#### 1. 资料来源\n"
    ]

    for i, res in enumerate(search_results, 1):
        title = res.get('title', '无标题')
        url = res.get('url', '无链接')
        raw_content = res.get('content', '')
        content_preview = raw_content[:300] + '...' if len(raw_content) > 300 else raw_content

        content.append(f"##### 来源 {i}\n")
        content.append(f"- **标题：** {title}\n")
        content.append(f"- **链接：** {url}\n")
        content.append(f"- **内容摘要：**\n")
        content.append(f"  ```\n  {content_preview}\n  ```\n")

    content.append("\n#### 2. 识别的问题\n")
    content.append(f"- 需要获取关于 **\"{section_title}\"** 的详细资料\n")
    content.append(f"- 搜索关键词：{topic} {section_title}\n")
    content.append(f"- 获取到 {len(search_results)} 条相关结果\n")

    content.append("\n#### 3. 生成的解决方案/输出\n")
    content.append("**格式化后的研究数据：**\n")
    content.append("```\n")
    content.append(formatted_data[:1000])
    if len(formatted_data) > 1000:
        content.append("\n... (内容已截断，完整内容见上方)")
    content.append("\n```\n")

    content.append("---\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(content)

    return log_path

def append_analyst_log(keyword: str, topic: str, section_title: str, research_data: str, draft: str, instruction: str = "") -> str:
    log_path = init_log_file(keyword, topic)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = [
        f"### Analyst 节点执行 | {timestamp}\n",
        f"**主题：** {topic}\n",
        f"**章节：** {section_title}\n",
        f"**操作：** 深度分析与章节撰写\n\n",
        "#### 1. 资料来源\n"
    ]

    if research_data:
        content.append("```\n")
        content.append(research_data[:1500])
        if len(research_data) > 1500:
            content.append("\n... (原始研究数据已截断)")
        content.append("\n```\n")
    else:
        content.append("- 无研究数据\n")

    content.append("\n#### 2. 识别的问题与分析过程\n")
    if instruction:
        content.append(f"- **Reviewer 反馈：** {instruction}\n")
        content.append("- 根据反馈进行修改重写\n")
    else:
        content.append(f"- 基于提供的研究数据撰写 **\"{section_title}\"** 章节\n")
    content.append("- 使用专业术语，保持逻辑严密\n")
    content.append("- 确保每条结论有数据支撑\n")

    content.append("\n#### 3. 生成的解决方案/输出\n")
    content.append("**生成的章节草稿：**\n")
    content.append("```markdown\n")
    content.append(_format_content_for_markdown(draft[:2000]))
    if len(draft) > 2000:
        content.append("\n... (草稿内容已截断)")
    content.append("\n```\n")

    content.append("---\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(content)

    return log_path

def append_reviewer_log(keyword: str, topic: str, section_title: str, draft: str, is_approved: bool, feedback: str, research_data: str) -> str:
    log_path = init_log_file(keyword, topic)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    status_text = "✅ 通过" if is_approved else "❌ 未通过"

    content = [
        f"### Reviewer 节点执行 | {timestamp}\n",
        f"**主题：** {topic}\n",
        f"**章节：** {section_title}\n",
        f"**操作：** 审查草稿质量\n\n",
        "#### 1. 资料来源\n"
    ]

    if research_data:
        content.append("```\n")
        content.append(research_data[:800])
        if len(research_data) > 800:
            content.append("\n... (原始研究数据已截断)")
        content.append("\n```\n")
    else:
        content.append("- 无研究数据\n")

    content.append("\n#### 2. 识别的问题与分析\n")
    content.append(f"- **审查项目：** 幻觉检测、数据支撑、逻辑矛盾\n")
    content.append(f"- **审查结果：** {status_text}\n")

    content.append("\n#### 3. 生成的解决方案/输出\n")
    content.append(f"**审查反馈：**\n")
    content.append("```\n")
    content.append(feedback)
    content.append("\n```\n")

    if not is_approved:
        content.append("\n**后续行动：** 将反馈发送给 Analyst 进行修改重写\n")
    else:
        content.append("\n**后续行动：** 草稿通过审核，进入 Editor 整合阶段\n")

    content.append("---\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(content)

    return log_path

def append_editor_log(keyword: str, topic: str, final_report: str, sections_count: int) -> str:
    log_path = init_log_file(keyword, topic)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = [
        f"### Editor 节点执行 | {timestamp}\n",
        f"**主题：** {topic}\n",
        f"**操作：** 整合所有章节生成最终报告\n\n",
        "#### 1. 资料来源\n",
        f"- 共整合 **{sections_count}** 个已通过审核的章节\n",
        "- 各章节内容来自 Analyst 生成、Reviewer 审核通过的定稿\n\n",
        "#### 2. 识别的问题\n",
        "- 检查各章节间的逻辑衔接与过渡\n",
        "- 处理可能存在的内容冲突\n",
        "- 确保报告结构完整、层次清晰\n\n",
        "#### 3. 生成的解决方案/输出\n",
        "**最终报告预览（前2000字符）：**\n",
        "```markdown\n",
        _format_content_for_markdown(final_report[:2000]),
        "\n```\n"
    ]

    if len(final_report) > 2000:
        content.append(f"\n*（报告总长度：{len(final_report)} 字符，完整内容请查看最终输出文件）*\n")

    content.append("---\n")

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(content)

    return log_path

def append_workflow_summary(keyword: str, topic: str, total_sections: int, revision_count: int, final_report_length: int) -> str:
    log_path = init_log_file(keyword, topic)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = [
        f"\n---\n\n",
        f"## 工作流执行总结 | {timestamp}\n\n",
        f"**主题：** {topic}\n\n",
        "### 执行统计\n",
        f"- 总章节数：{total_sections}\n",
        f"- 迭代修订次数：{revision_count}\n",
        f"- 最终报告长度：{final_report_length} 字符\n\n",
        "### 节点执行顺序\n",
        "1. **Planner** - 生成研究大纲\n",
        "2. **Researcher** - 联网检索资料\n",
        "3. **Analyst** - 撰写章节草稿\n",
        "4. **Reviewer** - 审核草稿质量\n",
        "5. **Editor** - 整合生成最终报告\n\n",
        "---\n\n",
        f"> 日志生成完毕 | 共 {total_sections} 个章节 | {revision_count} 次迭代修订\n"
    ]

    with open(log_path, 'a', encoding='utf-8') as f:
        f.writelines(content)

    return log_path