import os
import json
import logging
from typing import Dict, Any, List, Callable
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

from tavily import TavilyClient
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. 配置日志记录
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ToolManager")


# ==========================================
# 2. 模拟 MCP 思想的 ToolManager
# ==========================================
class ToolManager:
    """
    负责工具的注册、管理以及提供统一的执行接口。
    可以通过 get_tool_schemas 方法将所有工具转化为 LLM 可以理解的 JSON 格式。
    """
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, func: Callable, description: str, parameters: Dict[str, Any]):
        """注册工具并提供 JSON Schema 描述"""
        self._tools[name] = {
            "func": func,
            "description": description,
            "parameters": parameters
        }
        logger.info(f"注册工具: {name}")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """为 Agent 提供统一的 JSON 调用接口描述"""
        schemas = []
        for name, meta in self._tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": meta["description"],
                    "parameters": meta["parameters"]
                }
            })
        return schemas

    def execute_tool(self, name: str, kwargs_json: str) -> str:
        """统一执行接口，包含详细日志记录方便调试"""
        logger.info(f"尝试调用工具: {name} | 原始参数: {kwargs_json}")
        if name not in self._tools:
            error_msg = f"工具不存在: {name}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg}, ensure_ascii=False)

        try:
            # 兼容字符串 JSON 或 字典传入
            kwargs = json.loads(kwargs_json) if isinstance(kwargs_json, str) else kwargs_json
            
            logger.info(f"执行工具 '{name}' 开始...")
            result = self._tools[name]["func"](**kwargs)
            
            # 日志截断，防止结果过长刷屏
            res_str = str(result)
            preview = res_str[:200] + "..." if len(res_str) > 200 else res_str
            logger.info(f"执行工具 '{name}' 成功。返回数据预览: {preview}")
            
            return json.dumps({"status": "success", "data": result}, ensure_ascii=False)
        except Exception as e:
            logger.error(f"工具 {name} 执行失败: {str(e)}", exc_info=True)
            return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False)


# ==========================================
# 3. 具体的工具实现
# ==========================================
def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """使用 Tavily SDK 进行网络搜索"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("未设置 TAVILY_API_KEY，将返回模拟数据用于调试。")
        return [{"title": f"Mock Title for {query}", "url": "https://mock.com", "content": f"Mock content answering: {query}"}]
    
    client = TavilyClient(api_key=api_key)
    try:
        response = client.search(query=query, max_results=max_results)
        return response.get("results", [])
    except Exception as e:
        logger.error(f"Tavily 搜索异常: {str(e)}")
        raise

def paper_fetcher(topic: str, limit: int = 3) -> List[Dict[str, str]]:
    """模拟获取学术论文的工具"""
    logger.info(f"获取主题 '{topic}' 的相关论文...")
    # 此处可扩展为调用 Arxiv, Semantic Scholar 等 API
    return [
        {
            "title": f"{topic} 的前沿研究 {i+1}", 
            "authors": "Alice, Bob", 
            "abstract": f"这是一篇关于 {topic} 的模拟论文摘要。"
        }
        for i in range(limit)
    ]


# ==========================================
# 4. RAG 辅助函数 (基于 Milvus Lite)
# ==========================================
class MilvusRAGHelper:
    """
    使用 Milvus Lite 作为临时向量库，用于存储和检索长文本段落。
    """
    def __init__(self, db_path: str = "./milvus_local.db", embedding_model_name: str = "all-MiniLM-L6-v2"):
        # 使用本地文件作为 Milvus Lite 存储
        self.client = MilvusClient(db_path)

        self.model = None
        self.dim = 384
        try:
            # 使用轻量级 SentenceTransformer 模型
            logger.info(f"正在加载 Embedding 模型 ({embedding_model_name})...")
            self.model = SentenceTransformer(embedding_model_name)
            # 尽量从模型读取维度，读不到就回退到默认值
            if hasattr(self.model, "get_sentence_embedding_dimension"):
                self.dim = int(self.model.get_sentence_embedding_dimension() or self.dim)
            elif hasattr(self.model, "get_embedding_dimension"):
                self.dim = int(self.model.get_embedding_dimension() or self.dim)
        except Exception as e:
            # 常见原因：本地无模型且无法联网下载 / 环境缺少依赖
            logger.warning(f"Embedding 模型加载失败，将禁用 RAG 检索（原因: {e}）")

    def init_collection(self, collection_name: str = "research_data"):
        """初始化/重置集合"""
        if self.model is None:
            raise RuntimeError("Embedding 模型不可用，无法初始化向量库。")
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            dimension=self.dim
        )
        logger.info(f"Milvus 集合 '{collection_name}' 初始化完成。")

    def add_documents(self, collection_name: str, texts: List[str]):
        """将长文本段落向量化并存储"""
        if self.model is None:
            raise RuntimeError("Embedding 模型不可用，无法向量化文档。")
        if not texts:
            return
        
        logger.info(f"开始向量化并存储 {len(texts)} 条文档段落...")
        embeddings = self.model.encode(texts)
        
        data = []
        for i, text in enumerate(texts):
            data.append({
                "id": i,
                "vector": embeddings[i].tolist(),
                "text": text
            })
        
        res = self.client.insert(collection_name=collection_name, data=data)
        logger.info(f"成功存入 Milvus，插入实体数量: {res.get('insert_count', len(texts))}")

    def search(self, collection_name: str, query: str, top_k: int = 3) -> List[str]:
        """支持语义/关键词检索"""
        if self.model is None:
            raise RuntimeError("Embedding 模型不可用，无法执行向量检索。")
        logger.info(f"Milvus 检索关键词: '{query}'")
        query_vector = self.model.encode([query])[0].tolist()
        
        results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text"]
        )
        
        # 解析返回结果
        retrieved_texts = []
        if results and len(results) > 0:
            for hit in results[0]:
                retrieved_texts.append(hit["entity"]["text"])
                
        logger.info(f"检索完成，找到 {len(retrieved_texts)} 条相关段落。")
        return retrieved_texts


# ==========================================
# 5. 初始化全局 ToolManager 并注册工具
# ==========================================
tool_manager = ToolManager()

# 注册 web_search
tool_manager.register_tool(
    name="web_search",
    func=web_search,
    description="搜索互联网获取最新信息。使用 Tavily 引擎。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "max_results": {"type": "integer", "description": "最大返回结果数，默认 5"}
        },
        "required": ["query"]
    }
)

# 注册 paper_fetcher
tool_manager.register_tool(
    name="paper_fetcher",
    func=paper_fetcher,
    description="获取指定主题的学术论文。",
    parameters={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "论文主题"},
            "limit": {"type": "integer", "description": "获取数量，默认 3"}
        },
        "required": ["topic"]
    }
)

if __name__ == "__main__":
    print("\n--- ToolManager Schema 测试 ---")
    print(json.dumps(tool_manager.get_tool_schemas(), indent=2, ensure_ascii=False))

    print("\n--- ToolManager 执行测试 ---")
    tool_manager.execute_tool("web_search", '{"query": "LangGraph 教程", "max_results": 1}')
    tool_manager.execute_tool("paper_fetcher", '{"topic": "Multi-Agent System", "limit": 2}')

    print("\n--- RAG Helper 测试 ---")
    # 初始化时会自动下载轻量级模型 (如果尚未下载)
    rag = MilvusRAGHelper()
    rag.init_collection("test_col")
    rag.add_documents("test_col", [
        "LangGraph 是一个用于构建有状态多智能体应用的库。",
        "Milvus 是一款开源的向量数据库，支持海量向量数据的相似度检索。",
        "Tavily 是专为 AI Agent 打造的搜索引擎。"
    ])
    rag.search("test_col", "LangGraph 是什么？", top_k=1)
