import os
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


class RagasEvaluationError(RuntimeError):
    """Raised when RAGAS cannot run in the current environment."""


def _get_eval_model() -> str:
    if os.getenv("RAGAS_EVAL_MODEL"):
        return os.getenv("RAGAS_EVAL_MODEL", "")
    if os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_BASE"):
        return os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
    return (
        os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )


def _get_api_key() -> Optional[str]:
    return os.getenv("RAGAS_EVAL_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")


def _get_base_url() -> Optional[str]:
    if os.getenv("RAGAS_EVAL_API_BASE"):
        return os.getenv("RAGAS_EVAL_API_BASE")
    if os.getenv("DEEPSEEK_API_KEY"):
        return os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com/v1"
    return None


def _build_ragas_llm():
    api_key = _get_api_key()
    if not api_key or api_key == "your-deepseek-api-key":
        raise RagasEvaluationError(
            "缺少评估模型 API Key。请设置 RAGAS_EVAL_API_KEY、DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
        )

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RagasEvaluationError("缺少 openai 依赖，请安装 requirements.txt 中的依赖。") from exc

    client_kwargs = {"api_key": api_key}
    base_url = _get_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        from ragas.llms import llm_factory

        client = AsyncOpenAI(**client_kwargs)
        return llm_factory(_get_eval_model(), client=client)
    except (ImportError, TypeError):
        pass

    try:
        from ragas.llms.base import llm_factory

        client = AsyncOpenAI(**client_kwargs)
        return llm_factory(_get_eval_model(), client=client)
    except (ImportError, TypeError):
        pass

    try:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RagasEvaluationError("当前环境缺少可用的 RAGAS LLM wrapper。") from exc

    chat_kwargs = {
        "model": _get_eval_model(),
        "api_key": api_key,
        "temperature": 0,
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
        raise RagasEvaluationError("最终研报为空，无法评估 Faithfulness。")
    if not retrieved_contexts:
        raise RagasEvaluationError("没有可用于评估的 retrieved_contexts。")

    try:
        from ragas.metrics.collections import Faithfulness
    except ImportError:
        Faithfulness = None

    llm = _build_ragas_llm()

    if Faithfulness is not None:
        scorer = Faithfulness(llm=llm)
        result = await scorer.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        return float(getattr(result, "value", result))

    try:
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import Faithfulness as LegacyFaithfulness
    except ImportError as exc:
        raise RagasEvaluationError("缺少 RAGAS Faithfulness 指标，请确认 ragas 已正确安装。") from exc

    sample = SingleTurnSample(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    scorer = LegacyFaithfulness(llm=llm)
    return float(await scorer.single_turn_ascore(sample))
