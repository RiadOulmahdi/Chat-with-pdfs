from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

import time

from app.config import settings
from app.metrics import QueryMetrics, estimate_cost_usd, log_metrics
from app.vectorstore import get_vector_store

# A follow-up question like "what about in 2023?" only makes sense together with
# the previous turn, but the retriever has no memory of the conversation - it just
# does a similarity search on whatever string it's given. So before retrieving we
# ask the LLM to rewrite the question into something standalone/self-contained.
CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, rewrite it as a standalone question that can "
    "be understood without the chat history. Do not answer the question, only "
    "reformulate it if needed, otherwise return it unchanged."
)

ANSWER_SYSTEM_PROMPT = (
    "You are an assistant answering questions about a corpus of PDF documents. "
    "Use only the following retrieved context to answer the question. Each "
    "context chunk is tagged with a citation marker like [1]. Cite the markers "
    "that support each claim in your answer, e.g. 'Revenue grew 12% [2].' "
    "If the answer is not contained in the context, say you don't know instead "
    "of guessing.\n\nContext:\n{context}"
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict]
    metrics: QueryMetrics


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)


def _contextualize_question(question: str, chat_history: list[BaseMessage], llm: ChatOpenAI) -> str:
    if not chat_history:
        return question
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"chat_history": chat_history, "question": question})


def _format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        location = f"{source}, page {page}" if page is not None else source
        parts.append(f"[{i}] ({location})\n{doc.page_content}")
    return "\n\n".join(parts)


def _docs_to_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "marker": i,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
        }
        for i, doc in enumerate(docs, start=1)
    ]


def ask(question: str, chat_history: list[BaseMessage] | None = None) -> RagAnswer:
    chat_history = chat_history or []
    llm = _get_llm()
    retriever = get_vector_store().as_retriever(search_kwargs={"k": settings.retriever_k})

    # Time the retrieval and generation steps separately so the "Performance"
    # panel in the UI can show where the latency actually comes from.
    start_total = time.perf_counter()

    standalone_question = _contextualize_question(question, chat_history, llm)

    start_retrieval = time.perf_counter()
    docs = retriever.invoke(standalone_question)
    retrieval_time_s = time.perf_counter() - start_retrieval

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    chain = answer_prompt | llm

    start_generation = time.perf_counter()
    response: AIMessage = chain.invoke(
        {
            "context": _format_context(docs),
            "chat_history": chat_history,
            "question": question,
        }
    )
    generation_time_s = time.perf_counter() - start_generation

    total_time_s = time.perf_counter() - start_total

    usage = response.usage_metadata or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

    metrics = QueryMetrics(
        question=question,
        retrieval_time_s=retrieval_time_s,
        generation_time_s=generation_time_s,
        total_time_s=total_time_s,
        num_docs_retrieved=len(docs),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimate_cost_usd(settings.chat_model, prompt_tokens, completion_tokens),
    )
    log_metrics(metrics)

    return RagAnswer(answer=response.content, sources=_docs_to_sources(docs), metrics=metrics)


def to_lc_history(pairs: list[tuple[str, str]]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for human, ai in pairs:
        history.append(HumanMessage(content=human))
        history.append(AIMessage(content=ai))
    return history
