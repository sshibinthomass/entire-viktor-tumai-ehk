"""Full-trajectory effort metrics.

Each trajectory's DEEPEST logged request embeds every earlier item (the export
guarantees call i+1's input contains call i's input plus its output), so the
deepest call is the fullest picture of what actually happened. All metrics are
counted inside that input.

Token numbers are chars/4 estimates — the export has no usage field. Context
tokens include JSON structural overhead while generated tokens don't (fine
under ranking; stated in the writeup for raw-dollar quotes), and image
placeholders are counted at a nominal IMG_TOKENS each (the redacted data URL
is ~6 tokens; a real image is ~1k).
"""
import json
import re

# Word-boundary error detection with negation handling. Checked on the first
# 400 AND last 800 chars of each tool output (tracebacks END outputs), after
# (a) turning JSON-escaped whitespace like "\\n" into real separators (in the
# serialized output "…\\nTraceback" reads as one word and defeats \b), and
# (b) stripping negated phrases ("0 failed", "no errors", '"errors": []')
# that the old substring match counted as errors.
ESCAPES_RE = re.compile(r'\\+[ntr"]')
NEGATED_RE = re.compile(
    r"\b(?:0|no|zero|without)\s+(?:tool\s+)?(?:errors?|failures?|failed|exceptions?)\b"
    r"|\b\d+\s+passed,?\s*0\s+failed\b"
    r"|\berrors?\W{0,4}(?:\[\s*\]|0\b|none\b|null\b|false\b)", re.I)
ERROR_RE = re.compile(
    r"\berror(?:s|ed)?\b|\btraceback\b|\bexceptions?\b|\bfail(?:ed|ure)s?\b"
    r"|\bexit code:?\s*[1-9]\d*\b", re.I)


def is_error_output(out_text):
    head, tail = out_text[:400], out_text[-800:]
    window = head if tail in head else head + "\n" + tail
    window = ESCAPES_RE.sub(" ", window)
    return bool(ERROR_RE.search(NEGATED_RE.sub(" ", window)))


IMG_TOKENS = 1000  # assumed tokens per redacted image placeholder

MODEL_ITEM_TYPES = {"function_call", "custom_tool_call", "reasoning"}


def _txt(part):
    if isinstance(part, str):
        return part
    return part.get("text", "") if isinstance(part, dict) else ""


def _content(item):
    c = item.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(_txt(p) for p in c)
    return ""


def est_tokens(s):
    return len(s) // 4


def _is_model_item(item):
    return item.get("type") in MODEL_ITEM_TYPES or item.get("role") == "assistant"


def trajectory_metrics(deepest_req, n_logged_calls):
    """Effort counters from the deepest call's input."""
    items = deepest_req["input"]

    n_tool_calls = n_tool_errors = n_user_turns = n_assistant_msgs = 0
    n_reasoning = reasoning_tokens = gen_tokens = tool_output_tokens = 0
    n_llm_calls = n_images = 0
    tools_used, streak, max_streak, last_tool = set(), 0, 0, None
    prev_model_item = False

    for it in items:
        t = it.get("type")
        role = it.get("role")

        is_model = _is_model_item(it)
        if is_model and not prev_model_item:
            n_llm_calls += 1  # one contiguous run of model-emitted items = one response
        prev_model_item = is_model

        if t in ("function_call", "custom_tool_call"):
            n_tool_calls += 1
            name = it.get("name") or "custom"
            tools_used.add(name)
            streak = streak + 1 if name == last_tool else 1
            max_streak = max(max_streak, streak)
            last_tool = name
            payload = it.get("arguments") if t == "function_call" else it.get("input")
            if not isinstance(payload, str):
                payload = json.dumps(payload or "")
            gen_tokens += est_tokens(payload)
        elif t in ("function_call_output", "custom_tool_call_output"):
            out = json.dumps(it.get("output"))
            tool_output_tokens += est_tokens(out)
            if is_error_output(out):
                n_tool_errors += 1
        elif t == "reasoning":
            n_reasoning += 1
            reasoning_tokens += est_tokens(
                "\n".join(_txt(s) for s in it.get("summary") or []))
        elif role == "assistant":
            n_assistant_msgs += 1
            gen_tokens += est_tokens(_content(it))
            # a retry streak is same-tool calls in a row WITHIN one working
            # burst; an assistant message or user turn breaks the burst
            streak, last_tool = 0, None
        elif role == "user":
            n_user_turns += 1
            streak, last_tool = 0, None
        if role == "user" and isinstance(it.get("content"), list):
            n_images += sum(1 for p in it["content"]
                            if isinstance(p, dict) and p.get("type") == "input_image")

    # the deepest request itself still got one (unlogged) response
    n_llm_calls += 1

    return {
        "n_llm_calls": n_llm_calls,
        "n_tool_calls": n_tool_calls,
        "n_tool_errors": n_tool_errors,
        "n_assistant_msgs": n_assistant_msgs,
        "n_user_turns": n_user_turns,
        "n_distinct_tools": len(tools_used),
        "max_repeat_streak": max_streak,
        "n_reasoning_items": n_reasoning,
        "reasoning_tokens": reasoning_tokens,
        "gen_tokens": gen_tokens,
        "tool_output_tokens": tool_output_tokens,
        # image placeholders serialize to ~6 tokens but a real image is ~1k;
        # count them at IMG_TOKENS so image-heavy tasks aren't undercounted
        "context_tokens": est_tokens(json.dumps(items)) + n_images * IMG_TOKENS,
        "n_logged_calls": n_logged_calls,
    }
