from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

def build_faiss_retriever( chunks : list[dict], top_k : int = 5) -> HuggingFaceEmbeddings :
    """
    Builds a retriever using the provided FAISS index and chunks(metadata)
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Separate texts and metadatas
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Build FAISS vectorstore directly from texts
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)

    # Return a LangChain retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    return retriever, vectorstore

def save_faiss_retriever(vectorstore, path : str | Path) :
    """
    Saves the FAISS retriever to disk
    """
    vectorstore.save_local(path)

def load_faiss_retriever(path : str | Path, top_k : int = 5) -> HuggingFaceEmbeddings :
    """
    Loads the FAISS retriever from disk
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever, vectorstore