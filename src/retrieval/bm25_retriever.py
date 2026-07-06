import json
from pathlib import Path
from langchain_community.retrievers import BM25Retriever

BM25_STORE_PATH = "data/bm25_chunks.json"


def build_bm25_retriever(chunks: list[dict], top_k: int = 5):
    """
    Builds a BM25 retriever from the given chunks and persists the raw
    chunks to disk so the index can be rebuilt on reload (BM25Retriever
    is in-memory only, unlike Chroma).
    """
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    retriever = BM25Retriever.from_texts(texts=texts, metadatas=metadatas)
    retriever.k = top_k * 4

    _save_chunks(chunks)
    return retriever


def load_bm25_retriever(top_k: int = 5):
    """
    Rebuilds the BM25 retriever from persisted chunks. Returns None if
    no chunks have been persisted yet.
    """
    path = Path(BM25_STORE_PATH)
    if not path.exists():
        return None

    chunks = json.loads(path.read_text())
    return build_bm25_retriever(chunks, top_k=top_k)


def add_bm25_retriever(new_chunks: list[dict], top_k: int = 5):
    """
    Adds new chunks by merging with persisted chunks and rebuilding
    the BM25 index (no incremental add exists for BM25Retriever).
    """
    path = Path(BM25_STORE_PATH)
    existing = json.loads(path.read_text()) if path.exists() else []
    merged = existing + new_chunks
    return build_bm25_retriever(merged, top_k=top_k)


def _save_chunks(chunks: list[dict]):
    path = Path(BM25_STORE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks))