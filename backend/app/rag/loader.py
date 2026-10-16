from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parents[3]
PDF_PATH = BASE_DIR / "data" / "raw" / "handbook_k67.pdf"


def load_pdf(pdf_path: Path):
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
