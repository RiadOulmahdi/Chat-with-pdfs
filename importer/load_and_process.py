import sqlite3
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from app.config import settings
from app.vectorstore import get_vector_store

PDF_MAGIC = b"%PDF-"
DB_PATH = Path("ingestion_state.db")


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingested_files (
            path TEXT PRIMARY KEY,
            mtime REAL
        )
        """
    )
    conn.commit()
    return conn


def extract_chunks(pdf_path: Path, splitter: RecursiveCharacterTextSplitter) -> list[Document]:
    with open(pdf_path, "rb") as f:
        if f.read(len(PDF_MAGIC)) != PDF_MAGIC:
            return []

    reader = PdfReader(pdf_path)
    docs = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        for chunk_idx, chunk in enumerate(splitter.split_text(text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_idx,
                        "chunk_id": f"{pdf_path.name}::p{page_idx}::c{chunk_idx}",
                    },
                )
            )
    return docs


def main():
    conn = init_db(DB_PATH)
    vector_store = get_vector_store()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Load already-ingested files into memory
    done_files = {row[0]: row[1] for row in conn.execute("SELECT path, mtime FROM ingested_files")}

    pdf_files = list(Path(settings.pdf_source_dir).glob("*.pdf"))

    for pdf_path in pdf_files:
        rel_path = pdf_path.name
        mtime = pdf_path.stat().st_mtime

        # Skip if file was already ingested and has not changed
        if done_files.get(rel_path) == mtime:
            print(f"Skipping (already ingested): {rel_path}")
            continue

        print(f"Processing: {rel_path}")
        chunks = extract_chunks(pdf_path, splitter)

        if chunks:
            # Add to vector store
            ids = [doc.metadata["chunk_id"] for doc in chunks]
            vector_store.add_documents(chunks, ids=ids)

        # Record file as processed in SQLite
        conn.execute(
            "INSERT OR REPLACE INTO ingested_files (path, mtime) VALUES (?, ?)",
            (rel_path, mtime),
        )
        conn.commit()

    conn.close()
    print("Ingestion complete.")


if __name__ == "__main__":
    main()