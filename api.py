import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from main import app as graph_app
from models import ResearchState


api = FastAPI(
    title="DeepResearch-MAS API",
    description="FastAPI wrapper for the LangGraph multi-agent research workflow.",
    version="0.1.0",
)


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="Research topic or question")
    enable_ragas_evaluation: bool = Field(
        default=False,
        description="Whether to run RAGAS faithfulness evaluation after report generation",
    )
    enable_initial_draft_evaluation: bool = Field(
        default=False,
        description="Whether to assemble first section drafts into a report and run RAGAS faithfulness",
    )


class ResearchCreateResponse(BaseModel):
    task_id: str
    status: str


class TaskSnapshot(BaseModel):
    task_id: str
    status: str
    topic: str
    progress: List[str]
    final_report: Optional[str] = None
    faithfulness_score: Optional[float] = None
    faithfulness_error: str = ""
    initial_draft_report: Optional[str] = None
    initial_draft_faithfulness_score: Optional[float] = None
    initial_draft_faithfulness_error: str = ""
    error: Optional[str] = None
    created_at: str
    updated_at: str


TASKS: Dict[str, Dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _task_or_404(task_id: str) -> Dict[str, Any]:
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task


def _append_progress(task_id: str, message: str) -> None:
    task = TASKS[task_id]
    task["progress"].append(message)
    task["updated_at"] = _now()


def _node_message(node_name: str, state: Any) -> str:
    if node_name == "planner":
        sections = _state_get(state, "sections", {})
        return f"Planner completed. Sections: {list(sections.keys())}"
    if node_name == "researcher":
        return "Researcher completed. Search and RAG context are ready."
    if node_name == "analyst":
        return "Analyst completed. Section drafts were generated or revised."
    if node_name == "reviewer":
        revision_count = _state_get(state, "revision_count", 0)
        sections = _state_get(state, "sections", {})
        failed_sections = []
        for title, section in sections.items():
            is_approved = _state_get(section, "is_approved", False)
            if not is_approved:
                failed_sections.append(title)
        if failed_sections:
            return (
                "Reviewer completed. "
                f"Revision round {revision_count}; failed sections: {failed_sections}"
            )
        return "Reviewer completed. All sections passed review."
    if node_name == "editor":
        return "Editor completed. Final report was assembled."
    if node_name == "initial_draft_editor":
        return "Initial-draft editor completed. First-draft report was assembled."
    if node_name == "initial_draft_ragas_evaluator":
        score = _state_get(state, "initial_draft_faithfulness_score")
        error = _state_get(state, "initial_draft_faithfulness_error", "")
        if score is not None:
            return f"Initial-draft RAGAS evaluator completed. Faithfulness={score:.4f}"
        return f"Initial-draft RAGAS evaluator failed or skipped score generation: {error}"
    if node_name == "ragas_evaluator":
        score = _state_get(state, "faithfulness_score")
        error = _state_get(state, "faithfulness_error", "")
        if score is not None:
            return f"RAGAS evaluator completed. Faithfulness={score:.4f}"
        return f"RAGAS evaluator failed or skipped score generation: {error}"
    return f"{node_name} completed."


async def _run_research_task(task_id: str, request: ResearchRequest) -> None:
    task = TASKS[task_id]
    task["status"] = "running"
    task["updated_at"] = _now()
    _append_progress(task_id, "Task started.")

    try:
        initial_state = ResearchState(
            topic=request.topic.strip(),
            enable_ragas_evaluation=request.enable_ragas_evaluation,
            enable_initial_draft_evaluation=request.enable_initial_draft_evaluation,
        )
        final_state: Any = None

        async for event in graph_app.astream(initial_state):
            for node_name, state_update in event.items():
                final_state = state_update
                _append_progress(task_id, _node_message(node_name, state_update))

                final_report = _state_get(state_update, "final_report", "")
                if final_report:
                    task["final_report"] = final_report

                task["faithfulness_score"] = _state_get(
                    state_update,
                    "faithfulness_score",
                    task.get("faithfulness_score"),
                )
                task["faithfulness_error"] = _state_get(
                    state_update,
                    "faithfulness_error",
                    task.get("faithfulness_error", ""),
                )
                initial_draft_report = _state_get(state_update, "initial_draft_report", "")
                if initial_draft_report:
                    task["initial_draft_report"] = initial_draft_report
                task["initial_draft_faithfulness_score"] = _state_get(
                    state_update,
                    "initial_draft_faithfulness_score",
                    task.get("initial_draft_faithfulness_score"),
                )
                task["initial_draft_faithfulness_error"] = _state_get(
                    state_update,
                    "initial_draft_faithfulness_error",
                    task.get("initial_draft_faithfulness_error", ""),
                )

        if final_state is not None:
            task["final_report"] = _state_get(
                final_state,
                "final_report",
                task.get("final_report"),
            )
            task["faithfulness_score"] = _state_get(
                final_state,
                "faithfulness_score",
                task.get("faithfulness_score"),
            )
            task["faithfulness_error"] = _state_get(
                final_state,
                "faithfulness_error",
                task.get("faithfulness_error", ""),
            )
            task["initial_draft_report"] = _state_get(
                final_state,
                "initial_draft_report",
                task.get("initial_draft_report"),
            )
            task["initial_draft_faithfulness_score"] = _state_get(
                final_state,
                "initial_draft_faithfulness_score",
                task.get("initial_draft_faithfulness_score"),
            )
            task["initial_draft_faithfulness_error"] = _state_get(
                final_state,
                "initial_draft_faithfulness_error",
                task.get("initial_draft_faithfulness_error", ""),
            )

        task["status"] = "completed"
        _append_progress(task_id, "Task completed.")
    except Exception as exc:
        task["status"] = "failed"
        task["error"] = str(exc)
        _append_progress(task_id, f"Task failed: {exc}")
    finally:
        task["updated_at"] = _now()


@api.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@api.post("/research", response_model=ResearchCreateResponse)
async def create_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchCreateResponse:
    task_id = str(uuid4())
    now = _now()
    TASKS[task_id] = {
        "task_id": task_id,
        "status": "queued",
        "topic": request.topic.strip(),
        "progress": [],
        "final_report": None,
        "faithfulness_score": None,
        "faithfulness_error": "",
        "initial_draft_report": None,
        "initial_draft_faithfulness_score": None,
        "initial_draft_faithfulness_error": "",
        "error": None,
        "created_at": now,
        "updated_at": now,
    }
    background_tasks.add_task(_run_research_task, task_id, request)
    return ResearchCreateResponse(task_id=task_id, status="queued")


@api.get("/research/{task_id}", response_model=TaskSnapshot)
async def get_research(task_id: str) -> TaskSnapshot:
    return TaskSnapshot(**_task_or_404(task_id))


@api.get("/research/{task_id}/report")
async def get_report(task_id: str, as_markdown: bool = False):
    task = _task_or_404(task_id)
    report = task.get("final_report")
    if not report:
        return {
            "task_id": task_id,
            "status": task["status"],
            "final_report": None,
            "error": task.get("error"),
        }
    if as_markdown:
        return PlainTextResponse(report, media_type="text/markdown; charset=utf-8")
    return {
        "task_id": task_id,
        "status": task["status"],
        "final_report": report,
        "faithfulness_score": task.get("faithfulness_score"),
        "faithfulness_error": task.get("faithfulness_error", ""),
        "initial_draft_report": task.get("initial_draft_report"),
        "initial_draft_faithfulness_score": task.get("initial_draft_faithfulness_score"),
        "initial_draft_faithfulness_error": task.get("initial_draft_faithfulness_error", ""),
    }


@api.get("/research/{task_id}/events")
async def stream_events(task_id: str):
    _task_or_404(task_id)

    async def event_generator():
        sent = 0
        while True:
            task = TASKS.get(task_id)
            if not task:
                yield "event: error\ndata: task not found\n\n"
                break

            progress = task.get("progress", [])
            while sent < len(progress):
                yield f"event: progress\ndata: {progress[sent]}\n\n"
                sent += 1

            if task.get("status") in {"completed", "failed"}:
                yield f"event: status\ndata: {task['status']}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
