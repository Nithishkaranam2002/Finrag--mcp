from pathlib import Path
import pandas as pd
from pypdf import PdfReader
from docx import Document

def read_txt_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def read_pdf(path: Path) -> str:
    text = []
    reader = PdfReader(str(path))
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)

def read_csv(path: Path, max_rows: int = 100) -> str:
    df = pd.read_csv(path).head(max_rows)
    # compact textual form
    lines = [", ".join(map(str, df.columns.tolist()))]
    for _, row in df.iterrows():
        lines.append(", ".join(map(lambda x: str(x), row.tolist())))
    return "\n".join(lines)

def load_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in [".txt", ".md"]:
        return read_txt_md(path)
    if suf == ".pdf":
        return read_pdf(path)
    if suf in [".docx", ".doc"]:
        return read_docx(path)
    if suf == ".csv":
        return read_csv(path)
    # fallback: try text
    return read_txt_md(path)
