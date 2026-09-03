from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from app.metrics import read_recent_metrics
from app.rag_chain import ask, to_lc_history

app = FastAPI(title="Chat with PDFs")


class ChatRequest(BaseModel):
    question: str
    history: list[tuple[str, str]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    metrics: dict


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    result = ask(request.question, to_lc_history(request.history))
    return ChatResponse(
        answer=result.answer,
        sources=result.sources,
        metrics=result.metrics.__dict__,
    )


@app.get("/metrics/recent")
async def recent_metrics(limit: int = 50) -> list[dict]:
    return read_recent_metrics(limit)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
