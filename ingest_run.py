# TODO: run once locally to build chroma_db/
"""
ingest_run.py — Run this ONCE to build the ChromaDB vector store.
Usage: python ingest_run.py
"""
import sys
from src.ingestion import ingest_all_pdfs
from pathlib import Path
from src.config import PDF_DIR, CHROMA_DIR

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
if __name__ == "__main__":
    print(f"PDF source : {PDF_DIR}")
    print(f"ChromaDB   : {CHROMA_DIR}")
    total = ingest_all_pdfs()
    if total == 0:
        print("\nIngestion failed. Check logs above.")
        sys.exit(1)
    print(f"\n✅ Success! {total} chunks stored in ChromaDB.")
    print("Next step: python api.py")