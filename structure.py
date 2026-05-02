#!/usr/bin/env python3
"""
structure.py — Project Setup Script for CrimeScope-AI
------------------------------------------------------
Run this ONCE before starting the project.
It checks for required folders/files and creates anything that is missing.

Usage:
    python structure.py
"""

import os
import logging
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging Setup — clean, readable output in the terminal
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  |  %(levelname)-8s  |  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Project Root — always relative to THIS file, not wherever you run from
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Folders that must exist before anything else runs
# ---------------------------------------------------------------------------
REQUIRED_DIRS = [
    ROOT / "data" / "raw_pdfs",      # Drop the 17 legal-act PDFs here
    ROOT / "data" / "processed",      # Cleaned text lives here (auto-generated)
    ROOT / "chroma_db",               # Vector store — built by ingestion.py
    ROOT / "src",                     # Core business logic package
    ROOT / "tests",                   # Assignment asks for 5-10 test cases
    ROOT / "logs",                    # Optional: save run logs here
]


# ---------------------------------------------------------------------------
# Empty placeholder files that need to exist for Python imports to work
# ---------------------------------------------------------------------------
REQUIRED_FILES = {
    # src/ package marker — without this, "from src.config import ..." fails
    ROOT / "src" / "__init__.py": "",

    # Every module starts as an empty file; we fill them one by one
    ROOT / "src" / "config.py":    "# TODO: paths, model names, constants\n",
    ROOT / "src" / "schemas.py":   "# TODO: Pydantic output models\n",
    ROOT / "src" / "prompts.py":   "# TODO: system & human prompt templates\n",
    ROOT / "src" / "ingestion.py": "# TODO: PDF → chunk → embed → ChromaDB\n",
    ROOT / "src" / "retriever.py": "# TODO: MQR + MMR retrieval logic\n",
    ROOT / "src" / "chain.py":     "# TODO: full LangChain RAG pipeline\n",

    # Entry-point scripts
    ROOT / "ingest_run.py":        "# TODO: run once locally to build chroma_db/\n",
    ROOT / "api.py":               "# TODO: FastAPI server\n",
    ROOT / "app.py":               "# TODO: Streamlit UI\n",

    # Tests — the assignment requires 5-10 sample inputs with outputs
    ROOT / "tests" / "__init__.py": "",
    ROOT / "tests" / "test_cases.py": "# TODO: 5-10 crime scenario test cases\n",
}


# ---------------------------------------------------------------------------
# .gitignore — things we never want in the repo
# ---------------------------------------------------------------------------
GITIGNORE_PATH = ROOT / ".gitignore"

# These lines will be ADDED only if they are not already present
GITIGNORE_ENTRIES = [
    "# Python",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "",
    "# Virtual environment",
    "venv/",
    ".venv/",
    "",
    "# Secrets — never commit API keys",
    ".env",
    "",
    "# Raw PDFs can be large; share them separately",
    "data/raw_pdfs/",
    "",
    "# Logs",
    "logs/",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_dirs(dirs: list[Path]) -> None:
    """Create every folder in the list if it does not already exist."""
    for directory in dirs:
        if directory.exists():
            log.info("EXISTS   %s", directory.relative_to(ROOT))
        else:
            directory.mkdir(parents=True, exist_ok=True)
            log.info("CREATED  %s", directory.relative_to(ROOT))


def create_files(files: dict[Path, str]) -> None:
    """
    Create placeholder files only when they are missing.
    Files that already exist (even empty ones) are left untouched
    so we never overwrite real work.
    """
    for filepath, default_content in files.items():
        if filepath.exists():
            log.info("EXISTS   %s", filepath.relative_to(ROOT))
        else:
            filepath.write_text(default_content, encoding="utf-8")
            log.info("CREATED  %s", filepath.relative_to(ROOT))


def update_gitignore(path: Path, entries: list[str]) -> None:
    """
    Append missing lines to .gitignore.
    We read what is already there and only add what is not present,
    so running this script twice never produces duplicates.
    """
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lines = set(existing.splitlines())

    missing = [line for line in entries if line not in existing_lines]

    if not missing:
        log.info("EXISTS   .gitignore  (nothing new to add)")
        return

    # Append a blank separator then the new entries
    with path.open("a", encoding="utf-8") as f:
        f.write("\n# --- added by structure.py ---\n")
        f.write("\n".join(missing) + "\n")

    log.info("UPDATED  .gitignore  (+%d lines)", len(missing))


def print_tree(root: Path) -> None:
    """Print a simple visual tree so you can confirm the layout at a glance."""
    print("\n" + "=" * 55)
    print("  PROJECT STRUCTURE")
    print("=" * 55)
    for item in sorted(root.rglob("*")):
        # Skip hidden folders, venv, and the vector store (can be large)
        parts = item.parts
        if any(p.startswith(".") or p in ("venv", "__pycache__", "chroma_db") for p in parts):
            continue
        depth = len(item.relative_to(root).parts) - 1
        prefix = "    " * depth + ("📁 " if item.is_dir() else "📄 ")
        print(prefix + item.name)
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Starting CrimeScope-AI project setup...")
    log.info("Root: %s", ROOT)
    print()

    log.info("--- Step 1 / 3 : Creating folders ---")
    create_dirs(REQUIRED_DIRS)
    print()

    log.info("--- Step 2 / 3 : Creating placeholder files ---")
    create_files(REQUIRED_FILES)
    print()

    log.info("--- Step 3 / 3 : Updating .gitignore ---")
    update_gitignore(GITIGNORE_PATH, GITIGNORE_ENTRIES)
    print()

    print_tree(ROOT)

    log.info("Setup complete!")
    log.info("Next step → drop your 17 legal-act PDFs into:  data/raw_pdfs/")
    log.info("Then run:  python ingest_run.py")


if __name__ == "__main__":
    main()