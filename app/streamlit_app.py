import pandas as pd
import streamlit as st

from app.config import settings
from app.metrics import read_recent_metrics
from app.rag_chain import ask, to_lc_history
from app.vectorstore import get_vector_store

st.set_page_config(page_title="Chat with PDFs", layout="wide")


def get_collection_count() -> int:
    try:
        return get_vector_store()._collection.count()
    except Exception:
        return 0


with st.sidebar:
    st.header("Chat with PDFs")
    st.caption(f"Chat model: `{settings.chat_model}`")
    st.caption(f"Embedding model: `{settings.embedding_model}`")

    if not settings.openai_api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file.")

    chunk_count = get_collection_count()
    st.metric("Chunks indexed", chunk_count)
    if chunk_count == 0:
        st.warning(
            "The vector store is empty. Run:\n\n"
            "`uv run python -m importer.load_and_process --limit 50`\n\n"
            "to ingest a sample of PDFs first."
        )

    st.divider()
    st.subheader("Performance")
    recent = read_recent_metrics(limit=50)
    if recent:
        df = pd.DataFrame(recent)
        st.metric("Avg response time (s)", round(df["total_time_s"].mean(), 2))
        st.metric("Avg tokens/query", int(df["total_tokens"].mean()))
        st.metric("Est. cost so far ($)", round(df["estimated_cost_usd"].sum(), 4))
        st.line_chart(df[["retrieval_time_s", "generation_time_s", "total_time_s"]])
    else:
        st.caption("No queries logged yet.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    page = source.get("page")
                    location = f"{source['source']}, page {page}" if page is not None else source["source"]
                    st.markdown(f"[{source['marker']}] {location}")

question = st.chat_input("Ask a question about your PDFs...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history_pairs = [
        (st.session_state.messages[i]["content"], st.session_state.messages[i + 1]["content"])
        for i in range(0, len(st.session_state.messages) - 1, 2)
        if st.session_state.messages[i]["role"] == "user"
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(question, to_lc_history(history_pairs))
        st.markdown(result.answer)
        st.caption(
            f"retrieval {result.metrics.retrieval_time_s:.2f}s | "
            f"generation {result.metrics.generation_time_s:.2f}s | "
            f"total {result.metrics.total_time_s:.2f}s | "
            f"{result.metrics.total_tokens} tokens | "
            f"${result.metrics.estimated_cost_usd:.4f}"
        )
        if result.sources:
            with st.expander("Sources"):
                for source in result.sources:
                    page = source.get("page")
                    location = f"{source['source']}, page {page}" if page is not None else source["source"]
                    st.markdown(f"[{source['marker']}] {location}")

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": result.sources}
    )
