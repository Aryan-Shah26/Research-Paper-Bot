from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)

def embed_chunks(chunks : list[dict], model = SentenceTransformer) -> tuple[list,list[dict]] :
    """
    Generate chunks from given text
    """
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar = True)
    return embeddings, chunks