# DeepResearch-MAS

Multi-agent deep research/report generator built with LangGraph.

## Features
- Planner → Researcher → Analyst → Reviewer → Editor workflow (with revision loop)
- Parallel section processing (`asyncio.gather`)
- Streamlit UI with streaming node logs and Markdown export
- Web search via Tavily; optional RAG context via Milvus Lite + sentence-transformers (auto-fallback when embeddings aren’t available)

## Quickstart
1. Create virtualenv and install deps:
   - `pip install -r requirements.txt`
2. Configure env:
   - Copy `.env.example` to `.env` and fill keys
3. Run:
   - `streamlit run app.py`

## Project Layout
- `app.py`: Streamlit UI
- `main.py`: LangGraph workflow (planner/researcher/analyst/reviewer/editor)
- `agents.py`: Agent prompts + LLM calls
- `tools.py`: Search + RAG helper
- `models.py`: Pydantic state models
