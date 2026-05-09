import os
import json
import asyncio
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 如果使用 DeepSeek 官方 API，URL 通常是 https://api.deepseek.com/v1 或类似。
# 请根据您实际的 API 服务商配置 BASE_URL 和 API_KEY
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# 初始化 ChatOpenAI，指向 DeepSeek v4 模型
llm = ChatOpenAI(
    model="deepseek-chat",  # DeepSeek V3/V4 常用模型标识，具体视服务商而定
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
1. 报告必须包含自动生成的目录（Table of Contents, TOC）。
2. 内容排版要美观，层级清晰。
3. 在报告末尾，根据各章节中出现的数据引用（如 [数据源1]），提取并生成统一的“参考资料”列表。
4. 润色段落间的过渡，使整篇文章读起来流畅自然。"""

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

if __name__ == "__main__":
    print("Agent 模块已加载，可被 main.py 调用。")
