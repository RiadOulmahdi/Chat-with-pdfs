import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field

from app.config import settings

# Public per-token pricing (USD per 1M tokens), hardcoded for a rough cost
# estimate. Not billing-accurate - OpenAI is the source of truth for real spend.
_PRICE_PER_MILLION_TOKENS_USD = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "text-embedding-3-small": {"prompt": 0.02, "completion": 0.0},
}


@dataclass
class QueryMetrics:
    question: str
    retrieval_time_s: float
    generation_time_s: float
    total_time_s: float
    num_docs_retrieved: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = _PRICE_PER_MILLION_TOKENS_USD.get(model)
    if not pricing:
        return 0.0
    return (prompt_tokens / 1_000_000) * pricing["prompt"] + (completion_tokens / 1_000_000) * pricing["completion"]


def log_metrics(metrics: QueryMetrics) -> None:
    settings.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.metrics_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")


def read_recent_metrics(limit: int = 50) -> list[dict]:
    if not settings.metrics_log_path.exists():
        return []
    with open(settings.metrics_log_path, "r", encoding="utf-8") as f:
        last_lines = deque(f, maxlen=limit)
    return [json.loads(line) for line in last_lines]
