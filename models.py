from pydantic import BaseModel, Field
from typing import Dict, Optional

class SectionState(BaseModel):
    """单个章节的状态管理"""
    title: str = Field(description="章节标题")
    research_data: str = Field(default="", description="该章节的原始研究数据")
    rag_context: str = Field(default="", description="RAG 召回的关键上下文（用于压缩输入与提升相关性）")
    draft: str = Field(default="", description="Analyst 撰写的章节草稿")
    initial_draft: str = Field(default="", description="First Analyst draft snapshot before reviewer revision")
    critique: str = Field(default="", description="Reviewer 给出的大修指令")
    is_approved: bool = Field(default=False, description="是否通过了 Reviewer 的审查")

class ResearchState(BaseModel):
    """全局工作流状态管理"""
    topic: str = Field(description="研究主题")
    sections: Dict[str, SectionState] = Field(default_factory=dict, description="所有章节的字典映射")
    revision_count: int = Field(default=0, description="全局迭代次数")
    final_report: str = Field(default="", description="Editor 生成的最终报告")
    conflict_resolutions: str = Field(default="", description="冲突处理记录")
    enable_ragas_evaluation: bool = Field(default=False, description="是否在本次流程中运行 RAGAS 自动化评估")
    enable_initial_draft_evaluation: bool = Field(default=False, description="Whether to assemble and evaluate the first-draft report")
    faithfulness_score: Optional[float] = Field(default=None, description="RAGAS Faithfulness 指标分数")
    faithfulness_error: str = Field(default="", description="RAGAS Faithfulness 评估失败信息")
    initial_draft_report: str = Field(default="", description="Editor report assembled from first section drafts")
    initial_draft_faithfulness_score: Optional[float] = Field(default=None, description="RAGAS Faithfulness score for the first-draft report")
    initial_draft_faithfulness_error: str = Field(default="", description="RAGAS Faithfulness error for the first-draft report")
