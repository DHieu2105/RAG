from app.rag.from_scratch.chunker import structure_chunk


def test_structure_chunk() -> None:
    blocks = [
        {
            "article": 7,
            "clause": 1,
            "page_start": 16,
            "page_end": 16,
            "text": "A" * 300,
        },
        {
            "article": 7,
            "clause": 2,
            "page_start": 16,
            "page_end": 17,
            "text": "B" * 400,
        },
        {
            "article": 8,
            "clause": 1,
            "page_start": 18,
            "page_end": 18,
            "text": "C" * 200,
        },
    ]

    chunks = structure_chunk(blocks, max_size=1000)

    assert len(chunks) == 2
    assert chunks[0]["article"] == 7
    assert chunks[0]["clauses"] == [1, 2]
    assert chunks[0]["page_start"] == 16
    assert chunks[0]["page_end"] == 17
    assert chunks[0]["text"] == f"{'A' * 300}\n\n{'B' * 400}"
    assert chunks[1]["article"] == 8


def test_structure_chunk_respects_max_size() -> None:
    blocks = [
        {
            "article": 7,
            "clause": 1,
            "page_start": 16,
            "page_end": 16,
            "text": "A" * 600,
        },
        {
            "article": 7,
            "clause": 2,
            "page_start": 17,
            "page_end": 17,
            "text": "B" * 500,
        },
    ]

    chunks = structure_chunk(blocks, max_size=1000)

    assert len(chunks) == 2
    assert chunks[0]["clauses"] == [1]
    assert chunks[1]["clauses"] == [2]
    assert all(len(chunk["text"]) <= 1000 for chunk in chunks)


def test_structure_chunk_splits_oversized_clause() -> None:
    blocks = [
        {
            "article": 7,
            "clause": 1,
            "page_start": 16,
            "page_end": 17,
            "text": "A" * 1200,
        }
    ]

    chunks = structure_chunk(blocks, max_size=500)

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 500 for chunk in chunks)
    assert all(chunk["article"] == 7 for chunk in chunks)
    assert all(chunk["clauses"] == [1] for chunk in chunks)
    assert all(chunk["page_start"] == 16 for chunk in chunks)
    assert all(chunk["page_end"] == 17 for chunk in chunks)


def test_long_clause() -> None:
    blocks = [
        {
            "article": 7,
            "article_title": "Điều 7. Đăng ký học tập",
            "clause": 5,
            "page_start": 16,
            "page_end": 17,
            "text": "A" * 250,
        }
    ]

    chunks = structure_chunk(blocks, max_size=100, overlap=20)

    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk["text"]) <= 100
        assert chunk["article"] == 7
        assert chunk["article_title"] == "Điều 7. Đăng ký học tập"
        assert chunk["clauses"] == [5]
        assert chunk["page_start"] == 16
        assert chunk["page_end"] == 17


def test_structure_chunk_rejects_invalid_max_size() -> None:
    try:
        structure_chunk([], max_size=0)
    except ValueError:
        return

    raise AssertionError("Expected ValueError to be raised")


def test_structure_chunk_rejects_invalid_overlap() -> None:
    for overlap in (-1, 100):
        try:
            structure_chunk([], max_size=100, overlap=overlap)
        except ValueError:
            continue

        raise AssertionError("Expected ValueError to be raised")


if __name__ == "__main__":
    test_structure_chunk()
    test_structure_chunk_respects_max_size()
    test_structure_chunk_splits_oversized_clause()
    test_long_clause()
    test_structure_chunk_rejects_invalid_max_size()
    test_structure_chunk_rejects_invalid_overlap()
    print("All tests passed!")
