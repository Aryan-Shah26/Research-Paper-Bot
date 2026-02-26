import fitz
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

def parse_pdf(file_path : str | Path) -> list[dict] :
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

def parse_html(file_path : str | Path) -> list[dict] :
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

def parse_file(file_path : str | Path) -> list[dict] :
    file_path = Path(file_path)

    if file_path.suffix == ".pdf" :
        return parse_pdf(file_path)
    elif file_path.suffix == ".html" :
        return parse_html(file_path)
    else :
        raise ValueError("Unsupported Format :( ")