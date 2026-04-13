"""
pdf_processor.py
================
Handles PDF text extraction and text splitting into overlapping chunks.
"""

import fitz  # PyMuPDF
from typing import List


def extract_text_from_pdf(file) -> str:
    """
    Extract all plain text from an uploaded PDF file.

    Args:
        file: A file-like object (e.g. from Streamlit's file_uploader).

    Returns:
        A single string containing all extracted text.
    """
    # Read raw bytes from the uploaded file object
    pdf_bytes = file.read()

    # Open with PyMuPDF using the bytes buffer
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    extracted_pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")        # plain-text extraction
        if text.strip():
            extracted_pages.append(text)

    doc.close()

    # Join all pages with a newline separator
    full_text = "\n\n".join(extracted_pages)
    return full_text


def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    """
    Split a large text into smaller overlapping chunks for embedding.

    Overlapping ensures context isn't lost at chunk boundaries.

    Args:
        text:          The full document text.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to repeat between consecutive chunks.

    Returns:
        A list of text chunks (strings).
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Try to break at a sentence or word boundary for cleaner chunks
        if end < text_length:
            # Look for a nearby sentence ending (. ! ?)
            boundary = _find_sentence_boundary(text, end, window=100)
            if boundary:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - overlap) to create overlap
        start += chunk_size - chunk_overlap
        if start >= text_length:
            break

    return chunks


def _find_sentence_boundary(text: str, pos: int, window: int = 100) -> int | None:
    """
    Search backwards from `pos` within `window` chars for a sentence-ending
    punctuation mark followed by whitespace. Returns the position after it,
    or None if none found.
    """
    search_start = max(0, pos - window)
    snippet = text[search_start:pos]

    # Look for '. ', '! ', '? ' or newline from the end
    for i in range(len(snippet) - 1, -1, -1):
        if snippet[i] in ".!?\n" and (i + 1 >= len(snippet) or snippet[i + 1] in " \n"):
            return search_start + i + 1

    return None
