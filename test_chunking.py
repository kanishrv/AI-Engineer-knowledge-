import pytest
from app.chunking import split_text

def test_chunks_cover_text_and_overlap():
    text = "word " * 500
    chunks = split_text(text, 100, 20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert all(chunks)

def test_empty_and_invalid():
    assert split_text("   ") == []
    with pytest.raises(ValueError):
        split_text("hello", 10, 10)
