from src.documents import split_text


def test_split_text_respects_size_and_preserves_content():
    text = "First paragraph.\n\n" + ("word " * 80) + "\n\nLast paragraph."
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert "First paragraph." in chunks[0]
    assert "Last paragraph." in chunks[-1]


def test_split_text_rejects_invalid_overlap():
    try:
        split_text("text", chunk_size=100, overlap=100)
    except ValueError as exc:
        assert "greater than overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
