# TODO: PDF → chunk → embed → ChromaDB
"""
src/ingestion.py — PDF Ingestion Pipeline for CrimeScope-AI
-------------------------------------------------------------
This file runs ONCE locally to build the ChromaDB vector store.
Never runs on the server — the pre-built chroma_db/ folder is
committed to the repo and loaded directly on startup.

Pipeline:
  17 Legal PDFs
      │
      ▼
  Extract text (pypdf + pdfplumber fallback)
      │
      ▼
  Clean & preprocess text
      │
      ▼
  Split into chunks (section-aware)
      │
      ▼
  Generate embeddings (Google gemini-embedding-001)
      │
      ▼
  Save to ChromaDB (persisted to disk)

Run via:
  python ingest_run.py
"""

import logging
import re
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# PDF readers
import pypdf
import pdfplumber

# LangChain — updated imports for v0.2+
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import (
    PDF_DIR,
    CHROMA_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    GOOGLE_API_KEY,
)

# ---------------------------------------------------------------------------
# Logging — clean terminal output during ingestion
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  |  %(levelname)-8s  |  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ACT NAME MAPPING
# Maps PDF filename → official full act name (used as metadata)
# This metadata is stored in ChromaDB and helps the LLM cite acts correctly.
# Key   = lowercase PDF filename (without .pdf)
# Value = exact official name to show in output JSON
# ---------------------------------------------------------------------------
ACT_NAME_MAP = {
    "Bharatiya Nyaya Sanhita, 2023":                     "Bharatiya Nyaya Sanhita, 2023",
    "Bharatiya Nagarik Suraksha Sanhita, 2023":          "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "Bharatiya Sakshya Adhiniyam, 2023":                 "Bharatiya Sakshya Adhiniyam, 2023",
    "Indian Penal Code, 1860":                           "Indian Penal Code, 1860",
    "Code of Criminal Procedure, 1973":                  "Code of Criminal Procedure, 1973",
    "Indian Evidence Act, 1872":                         "Indian Evidence Act, 1872",
    "information_technology_act_2000":                  "Information Technology Act, 2000",
    "narcotic_drugs_psychotropic_substances_act_1985":  "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "prevention_of_corruption_act_1988":                "Prevention of Corruption Act, 1988",
    "pocso_act_2012":                                   "Protection of Children from Sexual Offences Act, 2012",
    "unlawful_activities_prevention_act_1967":          "Unlawful Activities (Prevention) Act, 1967",
    "dowry_prohibition_act_1961":                       "Dowry Prohibition Act, 1961",
    "juvenile_justice_act_2015":                        "Juvenile Justice (Care and Protection of Children) Act, 2015",
    "protection_of_women_domestic_violence_act_2005":   "Protection of Women from Domestic Violence Act, 2005",
    "motor_vehicles_act_1988":                          "Motor Vehicles Act, 1988",
    "arms_act_1959":                                    "Arms Act, 1959",
    "prevention_of_money_laundering_act_2002":          "Prevention of Money Laundering Act, 2002",
}


# ---------------------------------------------------------------------------
# STEP 1 — Extract text from a single PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: Path) -> Optional[str]:
    """
    Extract raw text from a PDF file.

    Strategy:
      1. Try pypdf first — it is fast and works for most PDFs
      2. If pypdf returns empty/garbage text, fall back to pdfplumber
         which is slower but handles complex layouts better

    Returns the full text as a single string, or None if both fail.
    """
    text = ""

    # --- Primary: pypdf ---
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        text = "\n".join(pages_text)
        log.debug("pypdf extracted %d chars from %s", len(text), pdf_path.name)
    except Exception as e:
        log.warning("pypdf failed for %s: %s", pdf_path.name, e)
        text = ""

    # --- Fallback: pdfplumber (if pypdf gave < 500 chars, it likely failed) ---
    if len(text.strip()) < 500:
        log.info("pypdf result too short, trying pdfplumber for %s", pdf_path.name)
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                pages_text = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n".join(pages_text)
            log.debug("pdfplumber extracted %d chars from %s", len(text), pdf_path.name)
        except Exception as e:
            log.error("pdfplumber also failed for %s: %s", pdf_path.name, e)
            return None

    return text if text.strip() else None


# ---------------------------------------------------------------------------
# STEP 2 — Clean extracted text
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Remove noise from extracted PDF text.

    Legal PDFs often have:
      - Page numbers ("Page 1 of 45")
      - Headers/footers repeated on every page
      - Extra whitespace and special characters
      - Weird unicode from PDF encoding

    We clean these so embeddings focus on actual legal content.
    """
    # Remove page number patterns like "Page 1", "- 23 -", "1 | P a g e"
    text = re.sub(r"\bPage\s+\d+\s*(of\s*\d+)?\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"-\s*\d+\s*-", "", text)
    text = re.sub(r"\d+\s*\|\s*P\s*a\s*g\s*e", "", text, flags=re.IGNORECASE)

    # Remove excessive whitespace (3+ newlines → 2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove non-printable characters (keep normal unicode for Hindi/Sanskrit terms)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Normalize spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# STEP 3 — Split text into chunks
# ---------------------------------------------------------------------------

def split_into_chunks(text: str, act_name: str, source_file: str) -> list[Document]:
    """
    Split the cleaned text into overlapping chunks and attach metadata.

    Why RecursiveCharacterTextSplitter?
      It tries to split on paragraph breaks (\n\n) first, then single
      newlines, then sentences, then words. This keeps legal sections
      together as much as possible — much better than naive token splits.

    Metadata stored per chunk:
      - source    : original PDF filename (for debugging)
      - act_name  : official act name (cited in output JSON)
      - chunk_id  : unique ID for deduplication in retriever.py

    The act_name in metadata is what appears as "act" in the final JSON output.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,          # 1200 chars — covers longest legal sections
        chunk_overlap=CHUNK_OVERLAP,    # 200 chars — preserves cross-section context
        separators=[
            "\n\n",    # Prefer splitting at paragraph breaks (between sections)
            "\n",       # Then at line breaks
            ". ",        # Then at sentence ends
            " ",         # Last resort: word boundary
        ],
        length_function=len,
    )

    raw_chunks = splitter.split_text(text)

    # Wrap each chunk in a LangChain Document with rich metadata
    documents = []
    for i, chunk in enumerate(raw_chunks):
        if len(chunk.strip()) < 50:
            # Skip tiny fragments — they add noise without value
            continue

        doc = Document(
            page_content=chunk,
            metadata={
                "source":    source_file,          # e.g., "indian_penal_code_1860.pdf"
                "act_name":  act_name,             # e.g., "Indian Penal Code, 1860"
                "chunk_id":  f"{source_file}_{i}", # Unique ID for deduplication
            }
        )
        documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# STEP 4 — Main ingestion function
# ---------------------------------------------------------------------------

def ingest_all_pdfs() -> int:
    """
    Master function: load all PDFs → clean → chunk → embed → save to ChromaDB.

    Returns the total number of chunks stored.
    """
    log.info("Starting ingestion pipeline...")
    log.info("PDF directory : %s", PDF_DIR)
    log.info("ChromaDB path : %s", CHROMA_DIR)

    # --- Find all PDFs ---
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        log.error("No PDFs found in %s — please add the 17 legal act PDFs first.", PDF_DIR)
        return 0

    log.info("Found %d PDF files", len(pdf_files))

    # --- Process each PDF ---
    all_documents: list[Document] = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="pdf"):
        # Map filename → official act name
        stem = pdf_path.stem.lower()
        act_name = ACT_NAME_MAP.get(stem)

        if not act_name:
            # If filename not in our map, use cleaned filename as fallback
            act_name = pdf_path.stem.replace("_", " ").title()
            log.warning(
                "No act name mapping for '%s' — using '%s'. "
                "Add it to ACT_NAME_MAP in ingestion.py.",
                pdf_path.name, act_name
            )

        log.info("Processing: %s → %s", pdf_path.name, act_name)

        # Step 1: Extract
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            log.error("Could not extract text from %s — skipping.", pdf_path.name)
            continue

        # Step 2: Clean
        clean = clean_text(raw_text)
        log.info("  Extracted %d chars after cleaning", len(clean))

        # Step 3: Chunk
        docs = split_into_chunks(clean, act_name, pdf_path.name)
        log.info("  Created %d chunks", len(docs))

        all_documents.extend(docs)

    if not all_documents:
        log.error("No documents to embed — ingestion failed.")
        return 0

    log.info("Total chunks across all PDFs: %d", len(all_documents))

    # --- Step 4: Embed + Save to ChromaDB ---
    log.info("Initializing Google embedding model: %s", EMBEDDING_MODEL)

    # embedding_model = GoogleGenerativeAIEmbeddings(
    #     model=EMBEDDING_MODEL,
    #     google_api_key=GOOGLE_API_KEY,
    # )
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

    log.info("Embedding and saving to ChromaDB... (this may take 3-5 minutes)")
    start_time = time.time()

    # Chroma.from_documents() embeds all chunks and saves to disk in one call
    # persist_directory tells ChromaDB where to save the files
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    elapsed = time.time() - start_time
    total_chunks = vectorstore._collection.count()

    log.info("ChromaDB saved to: %s", CHROMA_DIR)
    log.info("Total vectors stored: %d", total_chunks)
    log.info("Embedding time: %.1f seconds", elapsed)
    log.info("Ingestion complete! You can now run api.py")

    return total_chunks