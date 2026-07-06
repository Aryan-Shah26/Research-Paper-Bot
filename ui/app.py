import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

from src.ingestion.parser import parse_file
from src.ingestion.chunker import chunk
from src.retrieval.hybrid_retriever import build_hybrid_index, add_to_hybrid_index, hybrid_search, multi_query_hybrid_search, multi_paper_search
from src.retrieval.query_transform import generate_multi_queries
from src.generation.llm import load_llm, generate_answer
from src.generation.citation_check import extract_cited_claims, verify_citations, filter_hallucinated_citations

load_dotenv(override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

st.set_page_config(page_title="Research Paper Q&A", layout="wide")
st.title("Research Paper Q&A")

if "dense_retriever" not in st.session_state:
    st.session_state.dense_retriever = None
if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "sources" not in st.session_state:
    st.session_state.sources = []  # uploaded filenames, for multi-paper mode

with st.sidebar:
    st.header("Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload PDF or HTML file(s)", type=["pdf", "html"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Process Paper(s)"):
        with st.spinner("Processing..."):
            try:
                for uploaded_file in uploaded_files:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    pages = parse_file(tmp_path)
                    chunks = chunk(pages)

                    if st.session_state.dense_retriever is None:
                        dense_r, bm25_r, vs = build_hybrid_index(chunks)
                        st.session_state.dense_retriever = dense_r
                        st.session_state.bm25_retriever = bm25_r
                        st.session_state.vectorstore = vs
                    else:
                        bm25_r = add_to_hybrid_index(chunks, st.session_state.vectorstore)
                        st.session_state.bm25_retriever = bm25_r

                    st.session_state.sources.append(uploaded_file.name)
                    os.unlink(tmp_path)
                    st.success(f"Processed {uploaded_file.name}: {len(chunks)} chunks")

            except ValueError as e:
                st.error(f"Document error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

    st.divider()
    mode = st.radio("Query mode", ["Single-paper hybrid", "Multi-query", "Multi-paper compare"])
    verify_citations_toggle = st.checkbox("Verify citations", value=True)

question = st.text_input("Ask a question:")

if question:
    if st.session_state.dense_retriever is None:
        st.warning("Please upload and process a paper first.")
    else:
        with st.spinner("Thinking..."):
            try:
                client = load_llm(HF_TOKEN)
                dense_r, bm25_r = st.session_state.dense_retriever, st.session_state.bm25_retriever

                if mode == "Single-paper hybrid":
                    docs = hybrid_search(question, dense_r, bm25_r, top_k=5)
                elif mode == "Multi-query":
                    queries = generate_multi_queries(client, question, n=4)
                    docs = multi_query_hybrid_search(queries, dense_r, bm25_r, top_k=5)
                else:  # Multi-paper compare
                    docs = multi_paper_search(question, dense_r, bm25_r, st.session_state.sources, top_k_per_source=3)

                result = generate_answer(client=client, question=question, retrieved_docs=docs)
                answer = result["answer"]

                if verify_citations_toggle:
                    claims = extract_cited_claims(answer)
                    if claims:
                        verified = verify_citations(client, claims, docs)
                        answer = filter_hallucinated_citations(answer, verified)

                st.subheader("Answer:")
                st.write(answer)

                st.subheader("Sources:")
                for source in result["sources"]:
                    st.caption(source)

            except RuntimeError as e:
                st.error(f"Generation error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")