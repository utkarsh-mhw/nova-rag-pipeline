"""Extract heading-aware chunks from the local FDA PDF collection."""

import re
from glob import glob
from pathlib import Path

import pandas as pd
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "fda_pdfs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MIN_SEGMENT_LENGTH = 30
MAX_HEADING_LENGTH = 150
SECTION_RE = re.compile(r"^\W*(§+\s*\d+\.\d+)\s*(.*)$")
HEADING_RE = re.compile(r"^#{1,3}\s+(.+)$")


def chunk_pdfs(output_dir: Path, *, pdf_dir: Path = PDF_DIR) -> pd.DataFrame:
    """Write rag_chunks.pkl to a caller-selected directory and return its rows."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for path in sorted(glob(str(Path(pdf_dir) / "*.pdf"))):
        source = Path(path).name
        pages = pymupdf4llm.to_markdown(path, page_chunks=True)
        section = "(front matter)"
        for page_number, page in enumerate(pages, start=1):
            segments = []
            lines = []
            for line in page["text"].split("\n"):
                cfr_section = SECTION_RE.match(line)
                heading = HEADING_RE.match(line)
                if cfr_section or heading:
                    if lines:
                        segments.append((section, "\n".join(lines)))
                    if cfr_section:
                        title = f"{cfr_section.group(1)} {cfr_section.group(2)}"
                    else:
                        title = heading.group(1)
                    section = re.sub(r"[*#_`]", "", title).strip()[:MAX_HEADING_LENGTH]
                    lines = [line]
                else:
                    lines.append(line)
            if lines:
                segments.append((section, "\n".join(lines)))

            for section_name, text in segments:
                text = text.strip()
                if len(text) < MIN_SEGMENT_LENGTH:
                    continue
                parts = [text] if len(text) <= CHUNK_SIZE else splitter.split_text(text)
                for part in parts:
                    chunks.append(
                        {
                            "chunk_id": len(chunks),
                            "source": source,
                            "page": page_number,
                            "section": section_name,
                            "text": part,
                            "n_chars": len(part),
                        }
                    )

    chunks_df = pd.DataFrame(chunks)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_df.to_pickle(output_dir / "rag_chunks.pkl")
    print(len(chunks_df), "chunks")
    print(chunks_df.groupby("source").size())
    print(chunks_df["n_chars"].describe())
    return chunks_df
