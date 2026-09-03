# Chat with PDFs

A small RAG (retrieval-augmented generation) project: put some PDFs in a
folder, ask questions about them in a chat UI, and get answers with sources
cited. Made with LangChain, Chroma (local vector database) and OpenAI's
`gpt-4o-mini` + `text-embedding-3-small`.

This started as a school-ish side project to actually learn how RAG works
end to end, not just call a library function.

## How it works

1. `importer/load_and_process.py` reads every PDF in `data/pdfs`, pulls the
   text out page by page with `pypdf`, cuts it into ~1000-character chunks,
   and embeds each chunk with OpenAI's embedding model into a local Chroma
   database (`data/chroma`). It keeps track of which files are already done
   in a small SQLite file, so running it again only processes new/changed
   PDFs instead of redoing everything.
2. `app/rag_chain.py` takes a question, rewrites it into a standalone
   question if there's chat history (so follow-ups work), searches Chroma
   for the closest chunks, and asks the chat model to answer using *only*
   that context, with `[1]`, `[2]`... citations pointing back to the file
   and page it came from.
3. You talk to it through a Streamlit chat app (`app/streamlit_app.py`) or
   a small FastAPI (`app/server.py`).

## What it does (and doesn't) handle

- Normal text PDFs (reports, papers): works fine, this is what it was
  tested on.
- Tables: the text still gets extracted, but not as a table — rows and
  columns get flattened into plain text, so pulling one exact cell out of a
  table isn't reliable.
- Encrypted PDFs: works if there's no real password (just "owner"
  encryption).

Also, while testing, I found out that a chunk of the original PDF folder
(scraped from arXiv) wasn't actually PDFs — they were reCAPTCHA pages saved
with a `.pdf` extension because the scraper got blocked. So the ingestion
script checks the real file header (`%PDF-`) and skips anything that isn't
really a PDF, instead of crashing or indexing garbage.

## Does it scale to a lot of PDFs?

This was built with a big folder of PDFs in mind (originally ~45k arXiv
files, a few GB), so a few things were done on purpose for that:

- Files are processed one at a time, nothing loads the whole folder into
  memory, so RAM use doesn't grow with folder size.
- It's resumable: stop it, run it again later, and it skips whatever was
  already ingested (tested this by interrupting a run partway through).

Everything runs sequentially, so ingesting thousands of
files would be slow (no multi-threading yet). Fine for a handful or a few
hundred PDFs, would need those two fixes before trusting it on a huge messy
folder overnight.

## Metrics

Every question that's asked gets logged to `data/metrics/query_metrics.jsonl`
with: retrieval time, generation time, total time, tokens used, and a rough
cost estimate (from real token counts, not guessed). The Streamlit sidebar
shows a running average of all that plus a little chart, so you can see if
answers are slow because of the search step or the LLM step.

On the current small test corpus (9 PDFs, ~830 chunks): ingestion takes
about 11 seconds, and a typical question takes 1.5-4 seconds and costs a
fraction of a cent.

## Example questions I tried

**"What are the five key trends transforming the automotive industry?"**
> Electrification, autonomy, sharing, connectivity, and yearly updates,
> collectively referred to as "eascy" [5].

Sourced from `pwc-five-trends-transforming-the-automotive-industry.pdf`, 1.5s.

**"How does the cost of owning a battery electric vehicle in the UK compare to a petrol car?"**
> ...a used BEV was found to be £2,781 cheaper than its petrol equivalent
> across 80% of the models reviewed... with an average saving over five
> years of £5,317 for those charging at home [1]...

Sourced from `Electric-vehicles-uk-code-report-27-2-25.pdf`, 2.8s.

**Testing that it says "I don't know" — "What does this corpus say about the outcome of the 2022 FIFA World Cup?"**
> I don't know.

None of the PDFs are about football, so this is the right answer. The
retriever still returns its 5 closest chunks no matter what (it has no
concept of "nothing matches"), but the answer prompt tells the model to
only use the given context and admit it doesn't know rather than guess —
and it held up here instead of making something up from unrelated
automotive text. Worth re-testing this any time the prompt changes, since a
confident wrong answer is worse than an honest "I don't know."

## Screenshots
<img width="524" height="855" alt="metrics" src="https://github.com/user-attachments/assets/be0b5ccf-07cd-4f10-a6f2-f1348e250e53" />
<img width="1334" height="588" alt="chat" src="https://github.com/user-attachments/assets/946518e4-9b14-47ed-8954-94fc4f116885" />
<img width="1710" height="869" alt="app" src="https://github.com/user-attachments/assets/0b769bf4-e685-467b-8bab-9d06e77f0312" />


## Setup

```bash
uv sync
cp .env.example .env
# then put your OPENAI_API_KEY in .env
```

## Usage

```bash
# 1. put your PDFs in data/pdfs/ (flat folder, no subfolders for now)

# 2. ingest them (safe to re-run, skips files already done)
uv run ingest

# 3. chat
uv run streamlit run app/streamlit_app.py
# -> http://localhost:8501
```

There's also an API if you don't want the UI:

```bash
uv run uvicorn app.server:app --reload
```
`POST /chat`, `GET /metrics/recent`, `GET /health`.

Or with Docker: `docker compose up --build` runs the UI on `:8501` and the
API on `:8000`.

Settings (model names, chunk size, how many chunks to retrieve, etc.) all
live in `app/config.py` and can be overridden from `.env`.

## Things I'd still want to add

- Per-file error handling in ingestion (one bad PDF shouldn't kill the run)
- Multi-threaded ingestion for speed on large folders
- Better table extraction (`pdfplumber` or similar)
- OCR for scanned PDFs, if I ever need that
