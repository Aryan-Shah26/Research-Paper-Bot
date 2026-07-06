import fitz
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

IMAGE_DIR = "data/extracted_images"


def parse_pdf(file_path : str | Path) -> list[dict] :
    
    try :
        file_path = Path(file_path)
        pages = []
        doc = fitz.open(file_path)

        for page in doc :
            text = page.get_text()

            if not text.strip():
                continue

            pages.append({
                "text" : text,
                "metadata" : {
                    "source" : file_path.name,
                    "page" : page.number + 1
                }   
            })

        doc.close()
        return pages
    
    except Exception as e :
        raise ValueError(f"Error parsing the file '{Path(file_path).name}': {str(e)} ")


def extract_tables(file_path: str | Path) -> list[dict]:
    """
    Extracts tables as markdown-formatted text chunks, one per table,
    tagged with content_type=table so downstream retrieval/UI can
    distinguish them from prose chunks.
    """
    file_path = Path(file_path)
    doc = fitz.open(file_path)
    tables = []

    try:
        for page in doc:
            found = page.find_tables()
            for i, table in enumerate(found.tables):
                markdown = table.to_markdown()
                if not markdown.strip():
                    continue
                tables.append({
                    "text": markdown,
                    "metadata": {
                        "source": file_path.name,
                        "page": page.number + 1,
                        "content_type": "table",
                        "table_index": i,
                    },
                })
        return tables
    finally:
        doc.close()


def extract_images(file_path: str | Path) -> list[dict]:
    """
    Extracts embedded images to disk, returns metadata records (not
    chunk text) pointing at the saved file. Caller decides whether to
    caption/OCR them for retrieval.
    """
    file_path = Path(file_path)
    out_dir = Path(IMAGE_DIR) / file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(file_path)
    images = []

    try:
        for page in doc:
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base = doc.extract_image(xref)
                out_path = out_dir / f"page{page.number + 1}_img{img_index}.{base['ext']}"
                out_path.write_bytes(base["image"])
                images.append({
                    "path": str(out_path),
                    "source": file_path.name,
                    "page": page.number + 1,
                })
        return images
    finally:
        doc.close()


def parse_html(file_path : str | Path) -> list[dict] :
    
    try :
        file_path = Path(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

        return [{
                "text" : text,
                "metadata" : {
                    "source" : file_path.name,
                    "page" : 1
                }
        }]
    
    except Exception as e :
        raise ValueError(f"Error parsing the file '{Path(file_path).name}': {str(e)} ")

def parse_file(file_path : str | Path) -> list[dict] :
    file_path = Path(file_path)

    if file_path.suffix == ".pdf" :
        return parse_pdf(file_path)
    elif file_path.suffix == ".html" :
        return parse_html(file_path)
    else :
        raise ValueError("Unsupported Format :( ")