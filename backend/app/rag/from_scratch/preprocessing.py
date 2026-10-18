"""Basic text preprocessing utilities implemented from scratch."""

import re


ROMAN_NUMERAL_PATTERN = re.compile(
    r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})"
)


def normalize_line(line: str) -> str:
    """Trim a line and collapse consecutive whitespace to one space."""
    stripped_line = line.strip()
    words = stripped_line.split()
    return " ".join(words)


def is_footer(line: str) -> bool:
    """Return whether a line matches a page footer such as '14 * STSV'."""
    normalized_line = normalize_line(line)
    parts = normalized_line.split()

    if len(parts) != 3 or parts[1] != "*":
        return False

    number_first = parts[0].isdigit() and parts[2] == "STSV"
    number_last = parts[0] == "STSV" and parts[2].isdigit()
    return number_first or number_last


def is_article_heading(line: str) -> bool:
    """Return whether a line starts with an article label such as 'Điều 7.'."""
    normalized_line = normalize_line(line)
    parts = normalized_line.split()

    if len(parts) < 2 or parts[0] != "Điều":
        return False

    article_number = parts[1]
    if article_number.endswith("."):
        article_number = article_number[:-1]

    return article_number.isdigit()


def is_chapter_heading(line: str) -> bool:
    """Return whether a line starts with a Roman-numbered chapter label."""
    normalized_line = normalize_line(line)
    parts = normalized_line.split()

    if len(parts) < 2 or parts[0].casefold() != "chương":
        return False

    chapter_number = parts[1].removesuffix(".").removesuffix(":").upper()
    return bool(chapter_number) and bool(
        ROMAN_NUMERAL_PATTERN.fullmatch(chapter_number)
    )


def is_clause_start(line: str) -> bool:
    """Return whether a line starts with a numbered clause such as '5.'."""
    normalized_line = normalize_line(line)
    parts = normalized_line.split()

    if not parts:
        return False

    clause_label = parts[0]
    return clause_label.endswith(".") and clause_label[:-1].isdigit()


def merge_lines(lines: list[str]) -> str:
    """Normalize and join non-empty, non-footer lines into one paragraph."""
    merged: list[str] = []

    for line in lines:
        normalized = normalize_line(line)
        if not normalized or is_footer(normalized):
            continue
        merged.append(normalized)

    return " ".join(merged)


def preprocess_pages(pages: list[dict]) -> list[dict]:
    """Convert extracted PDF pages into article and clause-aware blocks."""
    blocks: list[dict] = []
    current_chapter_title: str | None = None
    current_article: int | None = None
    current_article_title: str | None = None
    current_clause: int | None = None
    current_lines: list[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None

    def save_current_block() -> None:
        nonlocal current_lines, current_page_start, current_page_end

        text = merge_lines(current_lines)
        if text:
            block = {
                "article": current_article,
                "article_title": current_article_title,
                "clause": current_clause,
                "page_start": current_page_start,
                "page_end": current_page_end,
                "text": text,
            }
            if current_chapter_title is not None:
                block["chapter_title"] = current_chapter_title
            blocks.append(block)

        current_lines = []
        current_page_start = None
        current_page_end = None

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]

        for raw_line in text.splitlines():
            line = normalize_line(raw_line)
            if not line or is_footer(line):
                continue

            if is_chapter_heading(line):
                save_current_block()
                current_chapter_title = line
                current_article = None
                current_article_title = None
                current_clause = None
                continue

            if is_article_heading(line):
                save_current_block()
                article_label = line.split()[1]
                current_article = int(article_label.removesuffix("."))
                current_article_title = line
                current_clause = None
                continue

            if is_clause_start(line):
                save_current_block()
                clause_label = line.split()[0]
                current_clause = int(clause_label[:-1])

            if not current_lines:
                current_page_start = page_number
            current_page_end = page_number
            current_lines.append(line)

    save_current_block()
    return blocks

