import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from .loader import PDF_PATH, load_pdf


BASE_DIR = Path(__file__).resolve().parents[3]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "handbook_k67.jsonl"
DEFAULT_CHUNK_SIZE = 1_000
DEFAULT_CHUNK_OVERLAP = 150
MIN_PAGE_TEXT_LENGTH = 20

_PAGE_HEADER_PATTERN = re.compile(
    r"^(?:\d+\s*\*\s*STSV|STSV\s*\*\s*\d+)$",
    re.IGNORECASE,
)
_DECORATIVE_LINE_PATTERN = re.compile(r"^[.·•_\-\s]{10,}$")
_LIST_ITEM_PATTERN = re.compile(
    r"^(?:[-+•❖]|\(\d+\)|\d+[.)]|[a-zđ][.)]|[IVXLCDM]+[.)])\s+",
    re.IGNORECASE,
)
_SECTION_PATTERN = re.compile(
    r"^(?:PHẦN|CHƯƠNG|ĐIỀU|MỤC|BƯỚC)\s+(?:\d+|[IVXLCDM]+)",
    re.IGNORECASE,
)
_SPACE_PATTERN = re.compile(r"[ \t\f\v]+")


def _normalize_line(line: str) -> str:
    line = line.replace("\u00ad", "").replace("\uf0b4", "×")
    return _SPACE_PATTERN.sub(" ", line).strip()


def _starts_new_paragraph(line: str) -> bool:
    letters = "".join(character for character in line if character.isalpha())
    is_short_heading = bool(letters) and letters.isupper() and len(line) <= 140
    return bool(
        _LIST_ITEM_PATTERN.match(line)
        or _SECTION_PATTERN.match(line)
        or is_short_heading
    )


def clean_page_text(text: str) -> str:
    """Normalize extracted PDF text while retaining document structure."""
    normalized_text = unicodedata.normalize("NFC", text)
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")

    paragraphs: list[str] = []
    current_lines: list[str] = []
    content_line_count = 0

    def flush_paragraph() -> None:
        if current_lines:
            paragraphs.append(" ".join(current_lines))
            current_lines.clear()

    for raw_line in normalized_text.split("\n"):
        line = _normalize_line(raw_line)

        if not line:
            flush_paragraph()
            continue
        if _PAGE_HEADER_PATTERN.fullmatch(line):
            continue
        if _DECORATIVE_LINE_PATTERN.fullmatch(line):
            continue
        if content_line_count == 0 and line.isdigit():
            continue

        if _starts_new_paragraph(line):
            flush_paragraph()
        current_lines.append(line)
        content_line_count += 1

    flush_paragraph()
    return "\n\n".join(paragraphs).strip()


def _find_chunk_end(text: str, start: int, max_chars: int) -> int:
    hard_end = min(start + max_chars, len(text))
    if hard_end == len(text):
        return hard_end

    minimum_end = start + max_chars // 2
    candidates = [
        text.rfind(separator, minimum_end, hard_end)
        for separator in ("\n\n", ". ", "; ", ": ", "\n", " ")
    ]
    best_break = max(candidates)
    return best_break + 1 if best_break >= minimum_end else hard_end


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text at natural boundaries with a character-based overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = _find_chunk_end(text, start, chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break

        next_start = max(end - chunk_overlap, start + 1)
        next_space = text.find(" ", next_start, min(next_start + 40, end))
        start = next_space + 1 if next_space != -1 else next_start

    return chunks


def preprocess_pages(
    pages: Iterable[dict],
    source: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Clean and chunk loaded PDF pages, preserving source metadata."""
    chunks: list[dict] = []
    source_stem = Path(source).stem

    for page in pages:
        page_number = page["page_number"]
        cleaned_text = clean_page_text(page["text"])
        if len(cleaned_text) < MIN_PAGE_TEXT_LENGTH:
            continue

        for chunk_index, chunk in enumerate(
            chunk_text(cleaned_text, chunk_size, chunk_overlap),
            start=1,
        ):
            chunks.append(
                {
                    "chunk_id": (
                        f"{source_stem}-p{page_number:03d}-c{chunk_index:03d}"
                    ),
                    "text": chunk,
                    "source": source,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )

    return chunks


def preprocess_pdf(
    pdf_path: Path = PDF_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    pages = load_pdf(pdf_path)
    return preprocess_pages(
        pages,
        source=pdf_path.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def save_chunks(chunks: Iterable[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_count = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for chunk in chunks:
            output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            chunk_count += 1

    return chunk_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and chunk a PDF document for the RAG pipeline."
    )
    parser.add_argument("--input", type=Path, default=PDF_PATH)
    parser.add_argument("--output", type=Path, default=PROCESSED_PATH)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = preprocess_pdf(args.input, args.chunk_size, args.chunk_overlap)
    chunk_count = save_chunks(chunks, args.output)
    print(f"Saved {chunk_count} chunks to {args.output}")


if __name__ == "__main__":
    main()