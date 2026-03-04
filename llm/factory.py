"""Universal LLM factory — returns LangChain-compatible chat models.

Two public functions:

* ``get_llm()`` — reads ``LLM_PROVIDER`` / ``LLM_MODEL`` from ``.env``.
  Used by **all regular agents** (analyst, script_writer, executor,
  self_healer, reporter, explorer).
* ``get_browser_use_llm()`` — reads ``BROWSER_USE_PROVIDER`` /
  ``BROWSER_USE_MODEL`` / ``BROWSER_USE_API_KEY`` from ``.env``.
  Returns a ``ChatOpenAI`` instance pointed at the vision-capable
  provider.  **Only** used by ``tools/browser_use_runner.py``.

Supported regular providers: groq, cerebras, lmstudio.

Usage::

    from llm.factory import get_llm, get_browser_use_llm

    llm = get_llm()                         # regular agents
    vision_llm = get_browser_use_llm()      # browser_use_runner only
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────
_DEFAULT_PROVIDER = "cerebras"
_DEFAULT_MODEL: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "lmstudio": "meta-llama-3.1-8b-instruct",
    "cerebras": "gpt-oss-120b",
}

_BU_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
}

_BU_DEFAULT_MODEL: dict[str, str] = {
    "groq": "meta-llama/llama-4-scout-17b-16e-instruct",
    "cerebras": "llama-3.3-70b",
    "lmstudio": "meta-llama-3.1-8b-instruct",
}


# =====================================================================
# get_llm  —  regular agents (analyst, script_writer, healer, etc.)
# =====================================================================

def get_llm(
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Build and return a LangChain chat model based on ``.env`` settings.

    Environment variables consumed
    ------------------------------
    LLM_PROVIDER : str
        One of ``groq``, ``lmstudio``, ``cerebras`` (default: ``cerebras``).
    LLM_MODEL : str
        Model name appropriate for the chosen provider.
    GROQ_API_KEY : str
        Required when ``LLM_PROVIDER=groq``.
    CEREBRAS_API_KEY : str
        Required when ``LLM_PROVIDER=cerebras``.

    Returns
    -------
    BaseChatModel
    """
    provider = os.getenv("LLM_PROVIDER", _DEFAULT_PROVIDER).strip().lower()

    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("GROQ_MODEL")
        or _DEFAULT_MODEL.get(provider, _DEFAULT_MODEL["cerebras"])
    )

    logger.info("LLM factory (get_llm): provider=%s  model=%s", provider, model)

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
            api_key="lm-studio",
        )

    # ── Cerebras (cloud, OpenAI-compatible) ─────────────────────────
    if provider == "cerebras":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("CEREBRAS_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set. Add it to your .env file."
            )
        base_url = os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
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


# =====================================================================
# get_browser_use_llm  —  ONLY used by tools/browser_use_runner.py
# =====================================================================

def get_browser_use_llm(temperature: float = 0.1) -> BaseChatModel:
    """Build a LangChain ``ChatOpenAI`` pointed at the Browser Use vision provider.

    Environment variables consumed
    ------------------------------
    BROWSER_USE_PROVIDER : str
        ``groq`` (default), ``cerebras``, ``lmstudio``.
    BROWSER_USE_MODEL : str
        Vision-capable model name.
    BROWSER_USE_API_KEY : str
        Falls back to ``GROQ_API_KEY`` / ``CEREBRAS_API_KEY`` when empty.

    Returns
    -------
    BaseChatModel
        A ``ChatOpenAI`` instance backed by the chosen vision provider.
    """
    from langchain_openai import ChatOpenAI

    provider = os.getenv("BROWSER_USE_PROVIDER", "groq").strip().lower()
    default_model = _BU_DEFAULT_MODEL.get(provider, _BU_DEFAULT_MODEL["groq"])
    model = os.getenv("BROWSER_USE_MODEL", default_model)

    api_key = os.getenv("BROWSER_USE_API_KEY", "")
    if not api_key:
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "")
        elif provider == "cerebras":
            api_key = os.getenv("CEREBRAS_API_KEY", "")
        elif provider == "lmstudio":
            api_key = "lm-studio"

    if not api_key and provider != "lmstudio":
        raise RuntimeError(
            f"No API key for Browser Use (provider={provider}). "
            f"Set BROWSER_USE_API_KEY or the matching provider key in .env."
        )

    base_url = os.getenv(
        "BROWSER_USE_BASE_URL",
        _BU_BASE_URLS.get(provider, _BU_BASE_URLS["groq"]),
    )

    logger.info(
        "LLM factory (get_browser_use_llm): provider=%s  model=%s  base_url=%s",
        provider, model, base_url,
    )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
