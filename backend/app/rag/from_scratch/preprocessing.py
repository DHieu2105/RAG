"""Basic text preprocessing utilities implemented from scratch."""


def normalize_line(line: str) -> str:
    """Trim a line and collapse consecutive whitespace to one space."""
    stripped_line = line.strip()
    words = stripped_line.split()
    return " ".join(words)

