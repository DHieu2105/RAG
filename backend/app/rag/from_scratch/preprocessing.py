"""Structure-aware text preprocessing for the student handbook."""

import re


ROMAN_NUMERAL_PATTERN = re.compile(
    r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})"
)
PART_HEADING_PATTERN = re.compile(
    r"^PHẦN\s+(\d+(?:\.\d+)*)[.:]?(?:\s+.*)?$",
    re.IGNORECASE,
)
CHAPTER_HEADING_PATTERN = re.compile(
    r"^CHƯƠNG\s+([IVXLCDM]+|\d+)[.:]?(?:\s+.*)?$",
    re.IGNORECASE,
)
ARTICLE_HEADING_PATTERN = re.compile(
    r"^ĐIỀU\s+(\d+)[.:]?(?:\s+.*)?$",
    re.IGNORECASE,
)
CLAUSE_START_PATTERN = re.compile(r"^(\d+)[.)](?:\s+|$)")
EXPLICIT_CLAUSE_PATTERN = re.compile(
    r"^KHOẢN\s+(\d+)[.:)]?(?:\s+|$)",
    re.IGNORECASE,
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


def _match_heading(pattern: re.Pattern[str], line: str) -> re.Match[str] | None:
    return pattern.fullmatch(normalize_line(line))


def is_part_heading(line: str) -> bool:
    """Return whether a line starts a numbered handbook part."""
    return _match_heading(PART_HEADING_PATTERN, line) is not None


def is_article_heading(line: str) -> bool:
    """Return whether a line starts with an article label such as 'Điều 7.'."""
    return _match_heading(ARTICLE_HEADING_PATTERN, line) is not None


def is_chapter_heading(line: str) -> bool:
    """Return whether a line starts with a valid chapter label."""
    match = _match_heading(CHAPTER_HEADING_PATTERN, line)
    if match is None:
        return False

    chapter_number = match.group(1).upper()
    return chapter_number.isdigit() or bool(
        ROMAN_NUMERAL_PATTERN.fullmatch(chapter_number)
    )


def clause_number(line: str) -> int | None:
    """Extract a clause number from '1.' or an explicit 'Khoản 1' label."""
    normalized_line = normalize_line(line)
    match = CLAUSE_START_PATTERN.match(normalized_line)
    if match is None:
        match = EXPLICIT_CLAUSE_PATTERN.match(normalized_line)
    return int(match.group(1)) if match is not None else None


def is_clause_start(line: str) -> bool:
    """Return whether a line starts a numbered or explicit clause."""
    return clause_number(line) is not None


def merge_lines(lines: list[str]) -> str:
    """Normalize and join non-empty, non-footer lines into one paragraph."""
    merged: list[str] = []

    for line in lines:
        normalized = normalize_line(line)
        if not normalized or is_footer(normalized):
            continue
        merged.append(normalized)

    return " ".join(merged)


def _part_number(line: str) -> str:
    match = _match_heading(PART_HEADING_PATTERN, line)
    if match is None:
        raise ValueError(f"Not a part heading: {line}")
    return match.group(1)


def _chapter_number(line: str) -> str:
    match = _match_heading(CHAPTER_HEADING_PATTERN, line)
    if match is None:
        raise ValueError(f"Not a chapter heading: {line}")
    return match.group(1).upper()


def _article_number(line: str) -> int:
    match = _match_heading(ARTICLE_HEADING_PATTERN, line)
    if match is None:
        raise ValueError(f"Not an article heading: {line}")
    return int(match.group(1))


def _is_uppercase_title(line: str) -> bool:
    letters = [character for character in line if character.isalpha()]
    return bool(letters) and line == line.upper()


def preprocess_pages(pages: list[dict]) -> list[dict]:
    """Convert extracted PDF pages into part/chapter/article/clause blocks."""
    blocks: list[dict] = []
    current_part: str | None = None
    current_part_title: str | None = None
    current_chapter: str | None = None
    current_chapter_title: str | None = None
    current_article: int | None = None
    current_article_title: str | None = None
    current_clause: int | None = None
    current_lines: list[str] = []
    current_page_start: int | None = None
    current_page_end: int | None = None
    awaiting_part_title = False
    awaiting_chapter_title = False

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
            if current_part is not None:
                block["part"] = current_part
            if current_part_title is not None:
                block["part_title"] = current_part_title
            if current_chapter is not None:
                block["chapter"] = current_chapter
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

            if is_part_heading(line):
                save_current_block()
                current_part = _part_number(line)
                current_part_title = line
                current_chapter = None
                current_chapter_title = None
                current_article = None
                current_article_title = None
                current_clause = None
                awaiting_part_title = True
                awaiting_chapter_title = False
                continue

            if is_chapter_heading(line):
                save_current_block()
                current_chapter = _chapter_number(line)
                current_chapter_title = line
                current_article = None
                current_article_title = None
                current_clause = None
                awaiting_part_title = False
                awaiting_chapter_title = True
                continue

            if is_article_heading(line):
                save_current_block()
                current_article = _article_number(line)
                current_article_title = line
                current_clause = None
                awaiting_part_title = False
                awaiting_chapter_title = False
                continue

            next_clause = clause_number(line)
            if next_clause is not None:
                save_current_block()
                current_clause = next_clause
                awaiting_part_title = False
                awaiting_chapter_title = False

            elif awaiting_part_title and _is_uppercase_title(line):
                current_part_title = f"{current_part_title} {line}"
                awaiting_part_title = False
                continue

            elif awaiting_chapter_title and _is_uppercase_title(line):
                current_chapter_title = f"{current_chapter_title} {line}"
                awaiting_chapter_title = False
                continue

            else:
                awaiting_part_title = False
                awaiting_chapter_title = False

            if not current_lines:
                current_page_start = page_number
            current_page_end = page_number
            current_lines.append(line)

    save_current_block()
    return blocks

