# 📄 Research Paper Q&A Agent

A RAG-powered (Retrieval Augmented Generation) chatbot that lets you upload research papers and ask questions about them. Built with LangChain, ChromaDB, HuggingFace, and Streamlit.

---

## 🚀 Features

- Upload PDF or HTML research papers
- Automatically parses, chunks, and indexes content
- Semantic search using sentence-transformers embeddings
- Answer generation using Llama 3.2 via HuggingFace Inference API
- Source citations with paper name and page number
- Persistent vector storage with ChromaDB
- Graceful error handling for API limits, bad files, and network issues

---

## 🏗️ Architecture

```
User uploads PDF/HTML
        ↓
parser.py       → Extracts text + metadata (source, page)
        ↓
chunker.py      → Splits into 512 token chunks with overlap
        ↓
retriever.py    → Embeds chunks + stores in ChromaDB (persisted to disk)
        ↓
User asks question
        ↓
retriever       → Finds top 5 semantically relevant chunks
        ↓
llm.py          → Sends context + question to Llama 3.2
        ↓
Streamlit UI    → Displays answer + source citations
```

---

## 📁 Project Structure

```
research-paper-qa/
├── src/
│   ├── ingestion/
│   │   ├── parser.py         # PDF and HTML parsing (PyMuPDF + BeautifulSoup)
│   │   └── chunker.py        # Text chunking (RecursiveCharacterTextSplitter)
│   ├── retrieval/
│   │   └── retriever.py      # ChromaDB vector store + LangChain retriever
│   └── generation/
│       ├── llm.py            # HuggingFace Inference API client
│       └── prompts.py        # Prompt templates
├── ui/
│   └── app.py                # Streamlit frontend
├── data/
│   └── chroma_db/            # Persisted vector store (auto-created)
├── config/
│   └── settings.py           # Pydantic settings
├── tests/
├── .env.example              # Environment variable template
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10 or 3.11
- A HuggingFace account with an API token

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/research-paper-qa.git
cd research-paper-qa
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your HuggingFace token:

```
HF_TOKEN=hf_your_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 5. Run the app

```bash
streamlit run src/ui/app.py
```

Open your browser at `http://localhost:8501`

---

## 📖 Usage

1. Upload a PDF or HTML research paper using the sidebar
2. Click **"Process Paper"** and wait for it to be indexed
3. Type your question in the text input
4. Get an answer with source citations

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | meta-llama/Llama-3.2-3B-Instruct (HuggingFace) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| RAG Framework | LangChain |
| PDF Parsing | PyMuPDF (fitz) |
| HTML Parsing | BeautifulSoup4 |

---

## ⚠️ Known Limitations

- **Images and graphs in PDFs are not processed** — only text is extracted. Figures are partially captured through their captions.
- **Scanned PDFs** (image-only) will return empty pages as they contain no extractable text.
- **HuggingFace free tier** has rate limits — if you hit them, wait a moment and retry.

---


## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
