# TODO: FastAPI server
"""
api.py — FastAPI Server for CrimeScope-AI
------------------------------------------
Exposes the RAG pipeline as a REST API.

Endpoints:
  POST /analyze-crime  — main endpoint (Streamlit calls this)
  GET  /health         — health check (Render uses this)
  GET  /docs           — auto Swagger UI (free from FastAPI)

Run locally:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from src.config import LLM_MODEL
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.schemas import CrimeQuery, AnalysisResponse
from src.chain import analyze_crime, _get_cached_retriever

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Lifespan — runs on startup to pre-load ChromaDB into memory
# Without this, the FIRST query would be slow (cold start)
# With this, ChromaDB is ready before the first request arrives
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Server starting — pre-loading ChromaDB and retriever...")
    _get_cached_retriever()   # Warms up the cache
    log.info("Retriever ready. Server is live.")
    yield
    log.info("Server shutting down.")


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="CrimeScope AI",
    description="RAG-based Indian Legal Advisor — identifies applicable laws from crime descriptions.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Quick health check — Render pings this to verify the server is alive."""
    return {"status": "ok", "service": "CrimeScope-AI"}


@app.post("/analyze-crime", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_crime_endpoint(query: CrimeQuery):
    """
    Main endpoint — analyze a crime description and return applicable laws.

    - Input  : free-text crime scene description (20–2000 chars)
    - Output : structured JSON with crime types + applicable legal sections
    """
    try:
        log.info("Request received — description: %s...", query.description[:80])

        result, elapsed, _ = analyze_crime(description=query.description)

        return AnalysisResponse(
            success=True,
            analysis=result,
            processing_time_seconds=elapsed,
            model_used=LLM_MODEL,
        )

    except Exception as e:
        log.error("Pipeline error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Local dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)