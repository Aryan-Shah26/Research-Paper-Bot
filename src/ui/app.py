import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from pathlib import Path
from src.ingestion.parser import parse_file
from src.ingestion.chunker import chunk
from src.retrieval.retriever import build_faiss_retriever, save_faiss_retriever, load_faiss_retriever
from src.generation.llm import load_llm, generate_answer
from dotenv import load_dotenv
import tempfile
import os

load_dotenv(override=True)   #Load environment variables from .env file, override if already set in system

FAISS_path = "data/faiss_index"
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Research Paper Q&A", layout="wide")
st.title("Research Paper Q&A")

# Session State init
if "retriever" not in st.session_state :
    st.session_state.retriever = None
if "vectorstore" not in st.session_state : 
    st.session_state.vectorstore = None

#Sidebar - File upload
with st.sidebar :
    st.header("Upload Paper")
    uploaded_file = st.file_uploader(
        "Upload a PDF or HTML file",
        type = ["pdf", "html"]
    )

    if uploaded_file and st.button("Process Paper") :
        with st.spinner("Processing...") :
            # Save uploaded file to a temporary location
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp :
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            #Pipeline
            pages = parse_file(tmp_path)
            chunks = chunk(pages)
            retriever, vectorstore = build_faiss_retriever(chunks)

            #Save to disk
            save_faiss_retriever(vectorstore, FAISS_path)

            #Store in session
            st.session_state.retriever = retriever
            st.session_state.vectorstore = vectorstore

            os.unlink(tmp_path)   #Delete temp file
            st.success(f"Processed {len(chunks)} successfully!")


#Main 
question = st.text_input("Ask a question about the paper:")

if question :
    if st.session_state.retriever is None :
        st.warning("Please upload and process a paper first.")

    else :
        with st.spinner("Thinking...") :
            client = load_llm(HF_TOKEN)
            retrived_docs = st.session_state.retriever.invoke(question)
            result = generate_answer(client=client, question=question, retrieved_docs=retrived_docs)

        st.subheader("Answer : ")
        st.write(result["answer"])

        st.subheader("Sources : ")
        for source in result["sources"] :
            st.caption(source)