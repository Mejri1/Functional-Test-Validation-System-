"""Universal LLM factory — returns a LangChain-compatible chat model.

Reads ``LLM_PROVIDER`` from ``.env`` (or the environment) and builds the
matching LangChain chat model.  Supported providers:

* **groq** — Groq cloud API via ``ChatGroq``
* **lmstudio** — local LM Studio server (OpenAI-compatible) via ``ChatOpenAI``
* **cerebras** — Cerebras cloud API (OpenAI-compatible) via ``ChatOpenAI``

Usage::

    from llm.factory import get_llm
    llm = get_llm()                  # uses .env defaults
    llm = get_llm(temperature=0.3)   # override temperature
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
_DEFAULT_PROVIDER = "groq"
_DEFAULT_MODEL: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "lmstudio": "meta-llama-3.1-8b-instruct",
    "cerebras": "llama-3.3-70b",
}


def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Build and return a LangChain chat model based on ``.env`` settings.

    Environment variables consumed
    ------------------------------
    LLM_PROVIDER : str
        One of ``groq``, ``lmstudio``, ``cerebras`` (default: ``groq``).
    LLM_MODEL : str
        Model name appropriate for the chosen provider.
        Falls back to ``GROQ_MODEL`` for backward compatibility.
    GROQ_API_KEY : str
        Required when ``LLM_PROVIDER=groq``.
    CEREBRAS_API_KEY : str
        Required when ``LLM_PROVIDER=cerebras``.

    Parameters
    ----------
    temperature : float
        Sampling temperature (default 0.1).
    max_tokens : int
        Maximum tokens in the response (default 4096).

    Returns
    -------
    BaseChatModel
        A ready-to-use LangChain chat model instance.
    """
    provider = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()

    # Model: LLM_MODEL takes precedence, then legacy GROQ_MODEL, then default
    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("GROQ_MODEL")
        or _DEFAULT_MODEL.get(provider, _DEFAULT_MODEL["groq"])
    )

    logger.info("LLM factory: provider=%s  model=%s", provider, model)

    # ── Groq ────────────────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file."
            )

        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    # ── LM Studio (local, OpenAI-compatible) ────────────────────────
    if provider == "lmstudio":
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key="lm-studio",  # LM Studio accepts any string
        )

    # ── Cerebras (cloud, OpenAI-compatible) ─────────────────────────
    if provider == "cerebras":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("CEREBRAS_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set. Add it to your .env file."
            )

        base_url = os.getenv(
            "CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1"
        )

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        f"Supported: groq, lmstudio, cerebras"
    )
