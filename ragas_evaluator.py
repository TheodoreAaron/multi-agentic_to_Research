import os
import re
import sys
import types
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


class RagasEvaluationError(RuntimeError):
    """Raised when RAGAS cannot run in the current environment."""


DEFAULT_EVAL_MAX_TOKENS = 8192
DEFAULT_RESPONSE_MAX_CHARS = 6000
DEFAULT_CONTEXT_MAX_CHARS = 6000
DEFAULT_CONTEXT_MAX_ITEMS = 8


def _install_ragas_vertexai_compat_shim() -> None:
    """
    RAGAS 0.4.3 imports langchain_community.chat_models.vertexai at module load
    time, but langchain-community 0.4.x no longer exposes that path. This project
    evaluates with OpenAI-compatible models, so a small placeholder is enough for
    RAGAS' type registry to import successfully.
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    try:
        __import__(module_name)
        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - import compatibility only
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def _ensure_ragas_importable() -> None:
    _install_ragas_vertexai_compat_shim()
    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise RagasEvaluationError(
            "Cannot import ragas in the current environment. Make sure requirements.txt "
            f"is installed in the same venv that runs the app. Original error: {exc}"
        ) from exc


def _get_eval_model() -> str:
    if os.getenv("RAGAS_EVAL_MODEL"):
        return os.getenv("RAGAS_EVAL_MODEL", "")
    if os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_BASE"):
        return os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
    return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


def _get_api_key() -> Optional[str]:
    return (
        os.getenv("RAGAS_EVAL_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def _get_base_url() -> Optional[str]:
    if os.getenv("RAGAS_EVAL_API_BASE"):
        return os.getenv("RAGAS_EVAL_API_BASE")
    if os.getenv("DEEPSEEK_API_KEY"):
        return os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1"
    return None


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        return default


def _get_eval_max_tokens() -> int:
    return _get_int_env("RAGAS_EVAL_MAX_TOKENS", DEFAULT_EVAL_MAX_TOKENS)


def _strip_references_section(text: str) -> str:
    pattern = r"\n+#{1,3}\s*(?:\u53c2\u8003\u8d44\u6599|\u53c2\u8003\u6587\u732e|References?)\s*\n.*$"
    return re.sub(pattern, "", text or "", flags=re.IGNORECASE | re.DOTALL).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0].strip()


def _prepare_evaluation_response(response: str) -> str:
    max_chars = _get_int_env("RAGAS_EVAL_RESPONSE_MAX_CHARS", DEFAULT_RESPONSE_MAX_CHARS)
    return _truncate_text(_strip_references_section(response), max_chars)


def _prepare_evaluation_contexts(retrieved_contexts: List[str]) -> List[str]:
    max_items = _get_int_env("RAGAS_EVAL_CONTEXT_MAX_ITEMS", DEFAULT_CONTEXT_MAX_ITEMS)
    max_total_chars = _get_int_env("RAGAS_EVAL_CONTEXT_MAX_CHARS", DEFAULT_CONTEXT_MAX_CHARS)
    if not retrieved_contexts:
        return []

    prepared: List[str] = []
    used_chars = 0
    for context in retrieved_contexts[:max_items]:
        remaining = max_total_chars - used_chars
        if remaining <= 0:
            break

        cleaned = _truncate_text(context, min(len(context), remaining))
        if cleaned:
            prepared.append(cleaned)
            used_chars += len(cleaned)
    return prepared


def _build_ragas_llm():
    api_key = _get_api_key()
    if not api_key or api_key == "your-deepseek-api-key":
        raise RagasEvaluationError(
            "Missing evaluation model API key. Set RAGAS_EVAL_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY."
        )

    _ensure_ragas_importable()

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RagasEvaluationError("Missing openai dependency. Install requirements.txt.") from exc

    client_kwargs = {"api_key": api_key}
    base_url = _get_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        from ragas.llms import llm_factory

        client = AsyncOpenAI(**client_kwargs)
        try:
            return llm_factory(
                _get_eval_model(),
                provider="openai",
                client=client,
                max_tokens=_get_eval_max_tokens(),
            )
        except TypeError:
            return llm_factory(_get_eval_model(), client=client, max_tokens=_get_eval_max_tokens())
    except (ImportError, TypeError):
        pass

    try:
        from ragas.llms.base import llm_factory

        client = AsyncOpenAI(**client_kwargs)
        try:
            return llm_factory(
                _get_eval_model(),
                provider="openai",
                client=client,
                max_tokens=_get_eval_max_tokens(),
            )
        except TypeError:
            return llm_factory(_get_eval_model(), client=client, max_tokens=_get_eval_max_tokens())
    except (ImportError, TypeError):
        pass

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RagasEvaluationError(
            "No usable RAGAS LLM wrapper is available. Confirm ragas, langchain-openai, "
            f"and langchain-community are installed correctly. Original error: {exc}"
        ) from exc

    chat_kwargs = {
        "model": _get_eval_model(),
        "api_key": api_key,
        "temperature": 0,
        "max_tokens": _get_eval_max_tokens(),
    }
    if base_url:
        chat_kwargs["base_url"] = base_url
    return LangchainLLMWrapper(ChatOpenAI(**chat_kwargs))


async def evaluate_faithfulness(
    user_input: str,
    response: str,
    retrieved_contexts: List[str],
) -> float:
    """
    Run RAGAS Faithfulness against a generated report.

    Uses the current RAGAS collections API first, with a legacy SingleTurnSample
    fallback for older installs.
    """
    if not response.strip():
        raise RagasEvaluationError("Final report is empty; cannot evaluate Faithfulness.")
    if not retrieved_contexts:
        raise RagasEvaluationError("No retrieved_contexts are available for evaluation.")

    evaluation_response = _prepare_evaluation_response(response)
    evaluation_contexts = _prepare_evaluation_contexts(retrieved_contexts)
    if not evaluation_response:
        raise RagasEvaluationError("Prepared evaluation response is empty; cannot evaluate Faithfulness.")
    if not evaluation_contexts:
        raise RagasEvaluationError("Prepared evaluation contexts are empty; cannot evaluate Faithfulness.")

    _ensure_ragas_importable()

    try:
        from ragas.metrics.collections import Faithfulness
    except ImportError:
        Faithfulness = None

    llm = _build_ragas_llm()

    if Faithfulness is not None:
        scorer = Faithfulness(llm=llm)
        result = await scorer.ascore(
            user_input=user_input,
            response=evaluation_response,
            retrieved_contexts=evaluation_contexts,
        )
        return float(getattr(result, "value", result))

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness as LegacyFaithfulness
    except ImportError as exc:
        raise RagasEvaluationError("Missing RAGAS Faithfulness metric. Confirm ragas is installed correctly.") from exc

    sample = SingleTurnSample(
        user_input=user_input,
        response=evaluation_response,
        retrieved_contexts=evaluation_contexts,
    )
    scorer = LegacyFaithfulness(llm=llm)
    return float(await scorer.single_turn_ascore(sample))
