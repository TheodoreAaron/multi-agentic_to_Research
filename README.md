# DeepResearch-MAS

## FastAPI API

This project can run the existing LangGraph workflow through a thin FastAPI
wrapper without changing the core agents.

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API service:

```bash
python run_api.py
```

Verify the FastAPI wrapper without importing heavy runtime dependencies:

```bash
python verify_api.py
```

After all project dependencies are installed, run the stricter import check:

```bash
python verify_api.py --runtime
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Main endpoints:

- `GET /health`: health check
- `POST /research`: submit a research task
- `GET /research/{task_id}`: inspect task status and progress
- `GET /research/{task_id}/report`: fetch the generated report
- `GET /research/{task_id}/events`: stream progress with Server-Sent Events

The first FastAPI version stores task state in memory. For production usage,
replace the in-memory task table with Redis or a database, and run long jobs
through a task queue such as Celery or RQ.

Multi-agent deep research/report generator built with LangGraph.

## Features
- Planner → Researcher → Analyst → Reviewer → Editor workflow (with revision loop)
- Parallel section processing (`asyncio.gather`)
- Streamlit UI with streaming node logs and Markdown export
- Web search via Tavily; optional RAG context via Milvus Lite + sentence-transformers (auto-fallback when embeddings aren’t available)
- Optional RAGAS Faithfulness evaluation for the final report

## Quickstart
1. Create virtualenv and install deps:
   - `pip install -r requirements.txt`
2. Configure env:
   - Copy `.env.example` to `.env` and fill keys
   - Optional RAGAS evaluation uses `RAGAS_EVAL_API_KEY` / `RAGAS_EVAL_API_BASE` / `RAGAS_EVAL_MODEL`, or falls back to the existing DeepSeek/OpenAI env vars
3. Run:
   - `streamlit run app.py`

## Project Layout
- `app.py`: Streamlit UI
- `main.py`: LangGraph workflow (planner/researcher/analyst/reviewer/editor)
- `ragas_evaluator.py`: RAGAS Faithfulness evaluation helper
- `agents.py`: Agent prompts + LLM calls
- `tools.py`: Search + RAG helper
- `models.py`: Pydantic state models
