from collections.abc import Callable

from app.rag.from_scratch.chunker import fixed_size_chunk


def assert_raises_value_error(function: Callable[[], object]) -> None:
    try:
        function()
    except ValueError:
        return

    raise AssertionError("Expected ValueError to be raised")


def test_fixed_size_chunk() -> None:
    chunks = fixed_size_chunk(
        "ABCDEFGHIJ",
        chunk_size=5,
        overlap=2,
    )

    assert chunks == ["ABCDE", "DEFGH", "GHIJ"]


def test_empty_text() -> None:
    assert fixed_size_chunk("", chunk_size=5, overlap=2) == []


def test_text_shorter_than_chunk_size() -> None:
    assert fixed_size_chunk("ABC", chunk_size=5, overlap=2) == ["ABC"]


def test_invalid_chunk_size() -> None:
    assert_raises_value_error(
        lambda: fixed_size_chunk("ABC", chunk_size=0, overlap=0)
    )


def test_invalid_overlap() -> None:
    assert_raises_value_error(
        lambda: fixed_size_chunk("ABC", chunk_size=5, overlap=-1)
    )
    assert_raises_value_error(
        lambda: fixed_size_chunk("ABC", chunk_size=5, overlap=5)
    )
    assert_raises_value_error(
        lambda: fixed_size_chunk("ABC", chunk_size=5, overlap=6)
    )


if __name__ == "__main__":
    test_fixed_size_chunk()
    test_empty_text()
    test_text_shorter_than_chunk_size()
    test_invalid_chunk_size()
    test_invalid_overlap()
    print("All tests passed!")
