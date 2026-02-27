import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from pathlib import Path
from src.ingestion.parser import parse_file
from src.ingestion.chunker import chunk
from src.retrieval.retriever import build_chroma_retriever, add_chroma_retriever, load_chroma_retriever,  rerank
from src.generation.llm import load_llm, generate_answer
from dotenv import load_dotenv
import tempfile
import os

load_dotenv(override=True)   #Load environment variables from .env file, override if already set in system

Chroma_path = "data/chroma_db"
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
    try :
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

                st.write(f"Pages found: {len(pages)}")
                st.write(f"Chunks found: {len(chunks)}")

                retriever, vectorstore = build_chroma_retriever(chunks)

                #Store in session
                st.session_state.retriever = retriever
                st.session_state.vectorstore = vectorstore

                os.unlink(tmp_path)   #Delete temp file
                st.success(f"Processed {len(chunks)} successfully!")
    
    except ValueError as e :
        st.error(f"Document error : {str(e)}")
    except RuntimeError as e :
        st.error(f"Runtime error : {str(e)}")
    except Exception as e :
        st.error(f"Unexpected error : {str(e)}")

    finally :
        if "tmp_path" in locals() and os.path.exists(tmp_path) :
            os.unlink(tmp_path)


#Main 
question = st.text_input("Ask a question about the paper:")

if question :
    if st.session_state.retriever is None :
        st.warning("Please upload and process a paper first.")

    else :
        with st.spinner("Thinking...") :
            try :
                client = load_llm(HF_TOKEN)
                retrived_docs = st.session_state.retriever.invoke(question)
                reranked_docs = rerank(query=question, docs=retrived_docs, top_k=5)
                result = generate_answer(client=client, question=question, retrieved_docs=reranked_docs)

                st.subheader("Answer : ")
                st.write(result["answer"])

                st.subheader("Sources : ")
                for source in result["sources"] :
                    st.caption(source)

            except RuntimeError as e:
                st.error(f"Generation error : {str(e)}")
            except Exception as e :
                st.error(f"Unexpected error during generation : {str(e)}")