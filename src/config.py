# TODO: paths, model names, constants
"""
src/config.py — Central Configuration for CrimeScope-AI
--------------------------------------------------------
This is the SINGLE SOURCE OF TRUTH for the entire project.
Every other file imports from here — nothing is hardcoded anywhere else.

Rule: If you want to change a model, a path, or a setting,
      you change it HERE and only here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file FIRST — before reading any os.getenv() calls
# Without this line, GOOGLE_API_KEY and LANGCHAIN_API_KEY will be None
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# Why Path(__file__)? Because __file__ = "C:/Users/.../CrimeScope-AI/src/config.py"
# .parent      = "C:/Users/.../CrimeScope-AI/src/"
# .parent.parent = "C:/Users/.../CrimeScope-AI/"   ← this is ROOT
# This way paths work on ANY machine — Windows, Mac, Linux, Render server
# # ---------------------------------------------------------------------------
# ROOT_DIR = Path(__file__).resolve().parent.parent

# # Folder where you drop the 17 legal act PDFs
# PDF_DIR = ROOT_DIR / "data" / "raw_pdfs"

# # Folder where ChromaDB saves its vector store files
# CHROMA_DIR = ROOT_DIR / "chroma_db"

# # Folder for saving log files (optional but professional)
# LOG_DIR = ROOT_DIR / "logs"



ROOT_DIR   = Path(__file__).resolve().parent.parent   # CrimeScope-AI/

PDF_DIR    = ROOT_DIR / "Data" / "raw_pdfs"
CHROMA_DIR = ROOT_DIR / "chroma_db"
LOG_DIR    = ROOT_DIR / "logs"

PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API KEYS — read from .env file (never hardcode keys in code!)
# os.getenv("KEY") reads the value from your .env file
# If key is missing, it returns None — we handle that gracefully below
# ---------------------------------------------------------------------------
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY")


# ---------------------------------------------------------------------------
# LLM MODEL CONFIGURATION
# Gemini 2.0 Flash = best balance of speed + quality (free tier available)
# To upgrade to Gemini 1.5 Pro later, change ONLY this one line
# ---------------------------------------------------------------------------
LLM_MODEL = "llama-3.3-70b-versatile"

# Temperature = how "creative" the LLM is
# 0.0 = deterministic, always same answer (best for legal/factual tasks)
# 1.0 = very creative (good for stories, bad for law)
LLM_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# EMBEDDING MODEL CONFIGURATION
# text-embedding-004 = Google's best free embedding model
# Used in ingestion.py to convert legal text → vectors
# Used in retriever.py to convert user query → vector for search
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# CHROMADB CONFIGURATION
# Collection = like a "table" in a database
# All 17 legal acts will be stored in this single collection
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = "legal_acts"


# ---------------------------------------------------------------------------
# RETRIEVAL CONFIGURATION
# These numbers directly affect answer quality — tune carefully
#
# TOP_K = how many chunks to retrieve from ChromaDB per sub-query
# Example: if MQR generates 4 sub-queries and TOP_K=5,
#          we get up to 20 chunks — MMR then picks best 10 unique ones
#
# MQR_QUERY_COUNT = how many sub-questions LLM generates from user query
# More = better recall, but slower. 4 is the sweet spot.
#
# MMR_FETCH_K = candidates MMR considers before picking final TOP_K
# Higher = more diversity in results (always >= TOP_K)
#
# MMR_LAMBDA = balance between relevance and diversity
# 0.0 = maximum diversity  |  1.0 = maximum relevance
# 0.5 = balanced (recommended)
# ---------------------------------------------------------------------------
# NEW — replace with:
MQR_LLM         = "llama-3.3-70b-versatile"  # Groq for fast sub-query generation
MQR_QUERY_COUNT = 4      # 4 sub-queries per user input
TOP_K           = 5      # ChromaDB results per sub-query
FINAL_TOP_K     = 12     # Chunks sent to Gemini after deduplication


# ---------------------------------------------------------------------------
# TEXT CHUNKING CONFIGURATION (used in ingestion.py)
# CHUNK_SIZE = max characters per chunk
# Legal sections can be long — 1000 chars keeps one section per chunk
#
# CHUNK_OVERLAP = how many characters two adjacent chunks share
# Overlap prevents important context from being cut off at chunk boundary
# Example: "...punishable under [CHUNK BREAK] Section 302 of IPC..."
#          With overlap, both chunks contain the connection
# ---------------------------------------------------------------------------
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 200


CHROMA_DB_PATH = str(CHROMA_DIR)
PDF_DIR_PATH   = str(PDF_DIR)

# ---------------------------------------------------------------------------
# LANGSMITH TRACING CONFIGURATION
# LangSmith tracks every LLM call, token count, latency, retrieved docs
# This is your MLOps observability layer — like logs but for AI chains
# LANGCHAIN_TRACING_V2=true must also be in your .env
# ---------------------------------------------------------------------------
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "CrimeScope-AI")


# ---------------------------------------------------------------------------
# FASTAPI CONFIGURATION
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"   # 0.0.0.0 = accept connections from any IP (needed for Render)
API_PORT = 8000


# ---------------------------------------------------------------------------
# VALIDATION — warn early if critical keys are missing
# This runs when any file does "from src.config import ..."
# Better to know NOW than to get a cryptic error deep inside a chain
# ---------------------------------------------------------------------------
import warnings

if not GOOGLE_API_KEY:
    warnings.warn(
        "GOOGLE_API_KEY not found in .env — Gemini LLM and embeddings will fail.",
        stacklevel=2
    )

if not LANGCHAIN_API_KEY:
    warnings.warn(
        "LANGCHAIN_API_KEY not found — LangSmith tracing will be disabled.",
        stacklevel=2
    )

if not GROQ_API_KEY:
    warnings.warn(
        "GROQ_API_KEY not found — Groq vector search will fail.",
        stacklevel=2
    )    