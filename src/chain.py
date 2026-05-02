
import logging
import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

from src.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from src.prompts import RAG_PROMPT
from src.schemas import CrimeAnalysis
from src.retriever import load_vectorstore, retrieve_legal_chunks, format_context

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache vectorstore (Chroma object) — loaded ONCE at startup
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_cached_vectorstore():
    """Load Chroma vectorstore once, cache forever."""
    log.info("Loading vectorstore into memory (first request only)...")
    vs = load_vectorstore()
    log.info("Vectorstore ready.")
    return vs


# Keep this name so api.py import doesn't break
def _get_cached_retriever():
    return _get_cached_vectorstore()


def _build_llm() -> ChatGroq:
    """Build and return a ChatGroq LLM with structured output."""
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing! "
            "Add it to your .env file: GROQ_API_KEY=gsk_xxxxxxxxxxxx"
        )

    log.info("Building LLM: llama-3.3-70b-versatile via Groq")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=LLM_TEMPERATURE,
        api_key=api_key,
    )
    return llm.with_structured_output(CrimeAnalysis)  # ← RETURN added (was missing!)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(min=1, max=4),
    reraise=True,
)
def analyze_crime(description: str) -> tuple[CrimeAnalysis, float, list]:
    """
    Full RAG pipeline:
      1. Retrieve relevant legal chunks from ChromaDB
      2. Format context
      3. Generate structured JSON analysis with Groq (llama-3.3-70b-versatile)
    """
    start = time.time()
    log.info("Pipeline start — query length: %d chars", len(description))

    # STEP 1: Retrieve
    log.info("Step 1/3 — Retrieving legal sections...")
    vectorstore = _get_cached_vectorstore()
    docs = retrieve_legal_chunks(description, vectorstore)
    if not docs:
        log.warning("No documents retrieved — ChromaDB may be empty.")

    # STEP 2: Format context
    context = format_context(docs)
    log.info("Step 2/3 — Context ready (%d chars, %d chunks)", len(context), len(docs))

    # STEP 3: Generate with Groq
    log.info("Step 3/3 — Generating legal analysis with llama-3.3-70b-versatile...")
    llm = _build_llm()

    messages = RAG_PROMPT.format_messages(
        context=context,
        question=description,
    )

    result: CrimeAnalysis = llm.invoke(messages)

    elapsed = round(time.time() - start, 2)
    log.info(
        "Pipeline complete in %.2fs — found %d laws",
        elapsed,
        len(result.applicable_laws),
    )

    return result, elapsed, docs