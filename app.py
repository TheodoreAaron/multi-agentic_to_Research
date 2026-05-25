import streamlit as st
import asyncio
from main import app as graph_app
from models import ResearchState
from logger import append_workflow_summary


def state_get(state, key, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


# 页面配置
st.set_page_config(page_title="DeepResearch-MAS", page_icon="🧠", layout="wide")

# ==========================================
# 侧边栏布局
# ==========================================
with st.sidebar:
    st.header("⚙️ 控制面板")
    st.write("欢迎使用 DeepResearch-MAS 多智能体系统！")
    
    topic = st.text_input("请输入研究主题 (Topic)", value="DeepResearch-MAS 架构分析")
    enable_ragas_evaluation = st.checkbox(
        "开启 RAGAS Faithfulness 评估",
        value=False,
        help="开启后会在本次调研流程生成最终研报后实际运行 RAGAS Faithfulness 评估；关闭时不会运行评估程序。"
    )
    enable_initial_draft_evaluation = st.checkbox(
        "开启初稿版报告 + RAGAS 评估",
        value=False,
        help="开启后会在 Analyst 首次生成各章节草稿后，让 Editor 另外组装一份初稿版报告并运行 RAGAS Faithfulness 评估。"
    )
    start_btn = st.button("🚀 开始生成", type="primary", use_container_width=True)
    
    st.divider()
    st.markdown("""
    **运行流程说明：**
    1. **Planner**: 动态大模型拆解提纲
    2. **Researcher**: 全网并发检索资料
    3. **Analyst (并行)**: 分章节撰写草稿
    4. **Reviewer (并行)**: 严苛审查打回
    5. **Editor**: 最终整合排版
    6. **RAGAS Evaluator (可选)**: Faithfulness 自动化评估
    """)

# ==========================================
# 异步执行函数 (用于流式捕获节点状态)
# ==========================================
async def run_workflow(
    topic: str,
    enable_ragas_evaluation: bool = False,
    enable_initial_draft_evaluation: bool = False,
):
    # 使用重构后的面向对象 State
    initial_state = ResearchState(
        topic=topic,
        enable_ragas_evaluation=enable_ragas_evaluation,
        enable_initial_draft_evaluation=enable_initial_draft_evaluation,
    )

    # 创建一个状态容器占位符
    status_container = st.empty()
    log_messages = []
    final_state = None

    # 使用 astream 流式获取节点输出
    async for event in graph_app.astream(initial_state):
        # event 的 key 通常是当前执行完毕的 node 名称
        for node_name, state_updates in event.items():
            final_state = state_updates
            
            # 根据不同的 node_name 更新状态展示
            if node_name == "planner":
                # state_updates 现在是一个 ResearchState 对象 (由于 LangGraph 的行为，可能是字典或对象)
                # 兼容字典和对象访问
                sections = state_get(state_updates, 'sections', {})
                msg = f"🗓️ **Planner** 节点执行完毕。动态生成大纲：{list(sections.keys())}"
            elif node_name == "researcher":
                msg = "🔍 **Researcher** 节点执行完毕。真实全网数据抓取完成！"
            elif node_name == "analyst":
                msg = "✍️ **Analyst** 节点执行完毕 (并行)。正在针对未通过章节进行撰写/重写..."
            elif node_name == "reviewer":
                rev_count = state_get(state_updates, 'revision_count', 0)
                sections = state_get(state_updates, 'sections', {})
                
                # 找出未通过的章节
                failed_sections = [title for title, sec in sections.items() if (hasattr(sec, 'is_approved') and not sec.is_approved) or (isinstance(sec, dict) and not sec.get('is_approved'))]
                
                if failed_sections:
                    msg = f"⚖️ **Reviewer** 审查完毕。发现问题章节：{failed_sections}，触发第 {rev_count} 次打回重写。"
                else:
                    msg = "✅ **Reviewer** 审查通过！所有章节数据支撑充分，无逻辑矛盾。"
            elif node_name == "editor":
                msg = "✨ **Editor** 节点执行完毕。最终长报告整合完成！"
            elif node_name == "initial_draft_editor":
                msg = "📝 **Initial Draft Editor** 节点执行完毕。初稿版报告已组装完成。"
            elif node_name == "initial_draft_ragas_evaluator":
                score = state_get(state_updates, 'initial_draft_faithfulness_score')
                error = state_get(state_updates, 'initial_draft_faithfulness_error', '')
                if score is not None:
                    msg = f"📊 **Initial Draft RAGAS Evaluator** 执行完毕。Faithfulness = {score:.4f}"
                else:
                    msg = f"⚠️ **Initial Draft RAGAS Evaluator** 执行失败：{error}"
            elif node_name == "ragas_evaluator":
                score = state_get(state_updates, 'faithfulness_score')
                error = state_get(state_updates, 'faithfulness_error', '')
                if score is not None:
                    msg = f"📊 **RAGAS Evaluator** 执行完毕。Faithfulness = {score:.4f}"
                else:
                    msg = f"⚠️ **RAGAS Evaluator** 执行失败：{error}"
            else:
                msg = f"🟢 **{node_name}** 执行完毕。"
            
            log_messages.append(msg)
            
            # 动态更新主界面的活动日志框
            with status_container.container():
                with st.status(f"当前状态流转：{node_name.capitalize()} 节点", expanded=True) as status:
                    for log in log_messages:
                        st.write(log)
                    if node_name == "ragas_evaluator" or (node_name == "editor" and not enable_ragas_evaluation):
                        status.update(label="🎉 报告生成完毕！", state="complete", expanded=False)
                    else:
                        status.update(label=f"正在运行... ({node_name})", state="running")
    
    return final_state

# ==========================================
# 主区域布局
# ==========================================
st.title("🧠 DeepResearch-MAS 控制台")

if start_btn:
    if not topic.strip():
        st.warning("请输入有效的研究主题！")
    else:
        st.subheader("1. 智能体活动日志 (Agent Activity Log)")
        
        # 运行异步图逻辑
        with st.spinner("系统初始化中..."):
            final_result = asyncio.run(
                run_workflow(
                    topic,
                    enable_ragas_evaluation,
                    enable_initial_draft_evaluation,
                )
            )

        report_md = ""
        if final_result:
            report_md = state_get(final_result, 'final_report', '')
            sections = state_get(final_result, 'sections', {})
            revision_count = state_get(final_result, 'revision_count', 0)
            faithfulness_score = state_get(final_result, 'faithfulness_score')
            faithfulness_error = state_get(final_result, 'faithfulness_error', '')
            initial_draft_report = state_get(final_result, 'initial_draft_report', '')
            initial_draft_faithfulness_score = state_get(final_result, 'initial_draft_faithfulness_score')
            initial_draft_faithfulness_error = state_get(final_result, 'initial_draft_faithfulness_error', '')

            try:
                append_workflow_summary(topic, topic, len(sections), revision_count, len(report_md))
                st.toast(f"节点日志已保存到 Log 文件夹（{topic}.md）", icon="📝")
            except Exception as e:
                st.warning(f"日志保存失败: {e}")
            
        if report_md:
            st.divider()
            st.subheader("2. 最终研报展示区")

            if enable_ragas_evaluation:
                if faithfulness_score is not None:
                    st.metric("RAGAS Faithfulness", f"{faithfulness_score:.4f}")
                else:
                    st.warning(f"RAGAS Faithfulness 评估未产出分数：{faithfulness_error}")

            if enable_initial_draft_evaluation and initial_draft_report:
                st.subheader("初稿版报告对照")
                if initial_draft_faithfulness_score is not None:
                    st.metric("Initial Draft RAGAS Faithfulness", f"{initial_draft_faithfulness_score:.4f}")
                else:
                    st.warning(f"初稿版 RAGAS Faithfulness 评估未产出分数：{initial_draft_faithfulness_error}")

                st.download_button(
                    label="下载初稿版 Markdown 报告",
                    data=initial_draft_report,
                    file_name="initial_draft_report.md",
                    mime="text/markdown"
                )

                try:
                    with open("initial_draft_report.md", "w", encoding="utf-8") as f:
                        f.write(initial_draft_report)
                except Exception as e:
                    st.warning(f"初稿版报告本地保存失败: {e}")

                with st.expander("查看初稿版报告", expanded=False):
                    st.markdown(initial_draft_report)
            
            # 下载按钮
            st.download_button(
                label="📥 下载 Markdown 报告",
                data=report_md,
                file_name="final_report.md",
                mime="text/markdown"
            )
            
            # 将生成的 Markdown 文件保存到本地 (接管原本 Editor 节点里的职责)
            try:
                with open("final_report.md", "w", encoding="utf-8") as f:
                    f.write(report_md)
                st.toast("最终报告已自动保存到本地根目录 `final_report.md`", icon="✅")
            except Exception as e:
                st.error(f"本地保存失败: {e}")
            
            # 使用 expander 或者直接展示
            with st.container(border=True):
                st.markdown(report_md)
else:
    st.info("👈 请在左侧输入主题并点击“开始生成”启动多智能体系统。")
