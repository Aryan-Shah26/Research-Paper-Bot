from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "data/chroma_db"

def get_embedder():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def build_chroma_retriever(chunks : list[dict], top_k : int = 5):
    """
    Builds a Chroma retriever from the given chunks.
    """

    embedder = get_embedder()
    
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vectorstores = Chroma.from_texts(
        texts= texts,
        embedding= embedder,
        metadatas= metadatas,
        persist_directory= CHROMA_PATH
    )

    retriever = vectorstores.as_retriever(search_kwargs={"k" : top_k})
    return retriever, vectorstores

def load_chroma_retriever(top_k : int = 5):
    """
    Loads a Chroma retriever from the persisted directory.
    """

    embedder = get_embedder()

    vectorstore = Chroma(
        persist_directory= CHROMA_PATH,
        embedding_function= embedder
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k" : top_k})
    return retriever,vectorstore

def add_chroma_retriever(chunks : list[dict], vectorstore : Chroma):
    """
    Adds new chunks to the existing Chroma retriever.
    """

    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    vectorstore.add_texts(
        texts= texts,
        metadatas= metadatas
    )