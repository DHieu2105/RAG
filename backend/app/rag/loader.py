from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def get_default_pdf_path() -> Path:
    standard_raw = RAW_DIR / "handbook_k67.pdf"
    if standard_raw.exists():
        return standard_raw

    raw_pdfs = list(RAW_DIR.glob("*.pdf"))
    if raw_pdfs:
        return raw_pdfs[0]

    processed_pdfs = list(PROCESSED_DIR.glob("*.pdf"))
    if processed_pdfs:
        return processed_pdfs[0]

    return standard_raw


PDF_PATH = get_default_pdf_path()


def load_pdf(pdf_path: Path = PDF_PATH):
    pages = []

    with pymupdf.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text")

            page_data = {
                "page_number": page_index + 1,
                "text": text,
            }
            pages.append(page_data)

    return pages


if __name__ == "__main__":
    result = load_pdf(PDF_PATH)
    print(f"Loaded {len(result)} pages from {PDF_PATH}")
