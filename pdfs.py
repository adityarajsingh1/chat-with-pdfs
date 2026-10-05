"""
Reading PDFs and cutting them into chunks.

Each chunk remembers which file and page it came from, so the answers
can point you to the right place ("report.pdf, p. 4").
"""

from pathlib import Path

from pypdf import PdfReader

# ~1000 characters is roughly a couple of paragraphs. small enough to be
# specific, big enough to still make sense on its own
CHUNK_SIZE = 1000
OVERLAP = 200


def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping pieces, trying to break at a sentence or space."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # look back a bit for a nicer place to cut
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def load_pdf(path):
    """Return a list of chunks: {"text", "source", "page"}."""
    path = Path(path)
    reader = PdfReader(path)
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for piece in split_text(text):
            chunks.append({"text": piece, "source": path.name, "page": page_num})
    return chunks


def load_folder(folder):
    """Load every PDF in a folder (or a single PDF if you pass a file)."""
    folder = Path(folder)
    files = [folder] if folder.is_file() else sorted(folder.glob("*.pdf"))
    chunks = []
    for f in files:
        print(f"reading {f.name}...")
        chunks.extend(load_pdf(f))
    return chunks
