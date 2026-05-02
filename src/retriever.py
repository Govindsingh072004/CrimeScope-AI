# # =============================================================================
# # src/retriever.py
# # =============================================================================

# import logging
# from typing import List

# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.documents import Document

# import src.config as cfg

# logger = logging.getLogger(__name__)

# from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, RETRIEVAL_TOP_K

# def get_embedding_model() -> HuggingFaceEmbeddings:
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )


# def load_vectorstore(embedding_model=None) -> Chroma:
#     if embedding_model is None:
#         embedding_model = get_embedding_model()

#     chroma_path = str(cfg.CHROMA_DIR)
#     logger.info(f"Loading ChromaDB from: {chroma_path}")

#     vectorstore = Chroma(
#         collection_name=cfg.CHROMA_COLLECTION_NAME,
#         embedding_function=embedding_model,
#         persist_directory=CHROMA_DB_PATH,
#     )
#     count = vectorstore._collection.count()
#     logger.info(f"ChromaDB loaded — {count} vectors available")
#     return vectorstore


# def get_retriever(vectorstore: Chroma, k: int = None):
#     k = k or cfg.FINAL_TOP_K
#     return vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": k, "fetch_k": k * 2},
#     )


# def retrieve_legal_chunks(query: str, vectorstore: Chroma, k: int = None) -> List[Document]:
#     k = k or cfg.FINAL_TOP_K
#     retriever = get_retriever(vectorstore, k=k)
#     docs = retriever.invoke(query)
#     logger.info(f"Retrieved {len(docs)} chunks for query: {query[:60]}...")
#     return docs


# def format_context(docs: List[Document]) -> str:
#     if not docs:
#         return "No relevant legal sections found."

#     context_parts = []
#     for i, doc in enumerate(docs, 1):
#         meta     = doc.metadata
#         act_name = meta.get("act_name", "Unknown Act")
#         section  = meta.get("section",  "Unknown Section")

#         context_parts.append(
#             f"[Source {i}] {act_name} | {section}\n"
#             f"{doc.page_content.strip()}"
#         )

#     return "\n\n---\n\n".join(context_parts)




# =============================================================================
# src/retriever.py
# =============================================================================

import logging
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import src.config as cfg

logger = logging.getLogger(__name__)

from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, FINAL_TOP_K as RETRIEVAL_TOP_K


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vectorstore(embedding_model=None) -> Chroma:
    if embedding_model is None:
        embedding_model = get_embedding_model()

    chroma_path = str(cfg.CHROMA_DIR)
    logger.info(f"Loading ChromaDB from: {chroma_path}")

    vectorstore = Chroma(
        collection_name=cfg.CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_DB_PATH,
    )
    count = vectorstore._collection.count()
    logger.info(f"ChromaDB loaded — {count} vectors available")
    return vectorstore


def get_retriever(vectorstore: Chroma, k: int = None):
    k = k or cfg.FINAL_TOP_K
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 2},
    )


def retrieve_legal_chunks(query: str, vectorstore: Chroma, k: int = None) -> List[Document]:
    k = k or cfg.FINAL_TOP_K
    retriever = get_retriever(vectorstore, k=k)
    docs = retriever.invoke(query)
    logger.info(f"Retrieved {len(docs)} chunks for query: {query[:60]}...")
    return docs


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "No relevant legal sections found."

    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta     = doc.metadata
        act_name = meta.get("act_name", "Unknown Act")
        section  = meta.get("section",  "Unknown Section")

        context_parts.append(
            f"[Source {i}] {act_name} | {section}\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(context_parts)
