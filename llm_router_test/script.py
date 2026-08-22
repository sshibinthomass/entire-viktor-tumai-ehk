"""
Naive cost-saving model router built on the `llm-router` package.

Architecture (matches the design diagram):

    request -> semantic difficulty classifier (llm-router, embeddings)
            -> small model  (cheap)  -> quality gate -> escalate on failure
            -> large model  (expensive)
            -> telemetry log (JSONL) for offline threshold tuning

Install:
    pip install llm-router chromadb sentence-transformers anthropic

Env:
    ANTHROPIC_API_KEY must be set.

Notes:
- llm-router matches a query against example sentences per route using
  embeddings (SentenceTransformer locally, or OpenAI embeddings).
- The classifier runs locally on a distilled embedding model, so routing
  adds only a few ms and no API cost per request.
- Anything the classifier can't confidently match falls through to the
  LARGE model. Fail expensive, not wrong.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field

import anthropic
from llm_router import Route, Router
from llm_router.chroma import SentenceTransformer

# ---------------------------------------------------------------------------
# 1. Route definitions: example sentences that *anchor* each difficulty tier.
#    These should come from real labeled traffic, not vibes. Start with a
#    handful, then continuously append misrouted examples from your logs.
# ---------------------------------------------------------------------------

ROUTES = [
    Route(
        name="small",
        sentences=[
            # short factual lookups
            "What is the capital of France?",
            "Convert 3 miles to kilometers",
            "What year did World War 2 end?",
            # formatting / mechanical transforms
            "Reformat this list as a markdown table",
            "Fix the grammar in this sentence",
            "Translate 'good morning' to Spanish",
            # simple classification / extraction
            "Is this review positive or negative?",
            "Extract the email addresses from this text",
            "Summarize this paragraph in one sentence",
        ],
    ),
    Route(
        name="large",
        sentences=[
            # multi-step reasoning
            "Prove that the square root of 2 is irrational",
            "Design a database schema for a multi-tenant SaaS app",
            "Debug this race condition in my concurrent code",
            # long-form generation with constraints
            "Write a detailed technical design doc for a payment system",
            "Refactor this module and explain every trade-off",
            # high-stakes / nuanced domains -> never downgrade
            "What do these lab results mean for my health?",
            "Review this contract clause for legal risk",
        ],
    ),
]

# threshold semantics in llm-router 0.1.1: Router.match() returns None
# ("no confident route") based on the engine's distance threshold. With the
# default of 0 everything matches its nearest route; raise it only after
# inspecting real distance values in your logs. Unmatched -> large model.
classifier = Router(ROUTES, SentenceTransformer(threshold=0))

# ---------------------------------------------------------------------------
# 2. Model tiers and dispatch
# ---------------------------------------------------------------------------

SMALL_MODEL = "claude-haiku-4-5-20251001"
LARGE_MODEL = "claude-sonnet-4-6"

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

# categories that must always bypass the router (tail-risk rule)
FORCE_LARGE_MARKERS = ("medical", "legal", "diagnos", "contract", "lawsuit")


def call_model(model: str, prompt: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


# ---------------------------------------------------------------------------
# 3. Quality gate: cheap, deterministic checks only. No LLM-as-judge on the
#    hot path (that would eat the savings) - sample offline instead.
# ---------------------------------------------------------------------------

REFUSAL_MARKERS = (
    "i can't help with that",
    "i'm not able to",
    "i don't have enough information",
)


def passes_quality_gate(prompt: str, response: str) -> bool:
    if not response or len(response.strip()) < 10:
        return False
    lowered = response.lower()
    if any(m in lowered for m in REFUSAL_MARKERS):
        return False
    # truncation heuristic: response ends mid-sentence
    if response.rstrip()[-1] not in ".!?`\")]}":
        return False
    return True


# ---------------------------------------------------------------------------
# 4. Telemetry: one JSONL line per request. This is the feedback loop -
#    offline evals replay these to tune routes, thresholds, and the gate.
# ---------------------------------------------------------------------------

LOG_PATH = os.environ.get("ROUTER_LOG", "router_log.jsonl")


def log_event(event: dict) -> None:
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------------------
# 5. The router itself
# ---------------------------------------------------------------------------

@dataclass
class RoutedResponse:
    text: str
    model_used: str
    route: str          # "small" | "large" | "forced_large" | "unmatched"
    escalated: bool
    latency_ms: float
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def handle(prompt: str) -> RoutedResponse:
    start = time.perf_counter()
    escalated = False

    # tail-risk bypass: never send high-stakes topics to the small model
    if any(m in prompt.lower() for m in FORCE_LARGE_MARKERS):
        route = "forced_large"
    else:
        route = classifier.match(prompt) or "unmatched"  # unmatched -> large

    if route == "small":
        text = call_model(SMALL_MODEL, prompt)
        if passes_quality_gate(prompt, text):
            model_used = SMALL_MODEL
        else:
            escalated = True
            text = call_model(LARGE_MODEL, prompt)
            model_used = LARGE_MODEL
    else:
        text = call_model(LARGE_MODEL, prompt)
        model_used = LARGE_MODEL

    latency_ms = (time.perf_counter() - start) * 1000
    result = RoutedResponse(text, model_used, route, escalated, latency_ms)

    log_event(
        {
            "request_id": result.request_id,
            "ts": time.time(),
            "route": route,
            "model_used": model_used,
            "escalated": escalated,
            "latency_ms": round(latency_ms, 1),
            "prompt_chars": len(prompt),
            # store the prompt hash, not the prompt, if privacy matters
            "prompt_preview": prompt[:120],
        }
    )
    return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for q in [
        "What's the capital of Australia?",
        "Summarize this in one line: The meeting covered Q3 budget overruns.",
        "Design a fault-tolerant event sourcing architecture for payments.",
        "What do these lab results mean for my health?",
    ]:
        r = handle(q)
        print(f"[{r.route:>12} -> {r.model_used}"
              f"{' (escalated)' if r.escalated else ''}] {q}")
        print(f"  {r.text[:100]}...\n")