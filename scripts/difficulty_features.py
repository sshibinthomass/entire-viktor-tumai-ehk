#!/usr/bin/env python3
"""Extract routing-time features + an observed-effort difficulty label.

Feature sources are restricted to what a router knows BEFORE dispatching a call:
  1. the `system` message
  2. the FIRST user message (Viktor's auto-read envelope + the trigger)
  3. the `tools` definitions

The export has no difficulty labels, so the label is a PROXY built from observed
effort inside the sampled call's history (model-generated tokens + tool-call
count), rank-averaged and cut into deciles 1..10. Read LABEL CAVEATS below
before quoting any number from this.

LABEL CAVEATS
  - Effort is not difficulty. A task can be long and easy, or short and hard.
  - Effort is model-dependent: a weaker model may burn more calls on the same
    task, so the label partly encodes the logged model. Measured skew is small
    but non-zero (gpt-family sits ~1 decile above claude-family).
  - The export samples ~1 call per task at a random point mid-task, so history
    size is a NOISY estimate of total task effort, not a measurement.

LEAKAGE
  The file-tool shape perfectly identifies the provider family
  (bash/file_* => claude, shell_command/apply_patch => gpt), as do `reasoning`
  items. Those features are dropped unless --allow-leaky is passed.

Usage:
    python scripts/difficulty_features.py export_linked/ -o results/features.jsonl
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SYS_TAGS = ["available_skills", "slack_history", "teams_history", "structured_output",
            "voice", "personalization", "custom_instructions", "personality",
            "thread_instructions", "core_philosophy", "operating_rules", "work_approach"]

USR_TAGS = ["auto_read_learnings", "auto_read_execution_log", "auto_read_channel_instructions",
            "auto_read_personal_skill", "auto_read_recent_activity", "delivery_note",
            "summary_so_far"]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SKILL_RE = re.compile(r"^\-\s\*\*([a-zA-Z0-9_\- ]+)\*\*", re.M)
HDR_RE = re.compile(r"\[((?:Slack|Microsoft Teams|Email|Cron|Scheduled|New)[^\]\n]{0,160})\]")
DAY_RE = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b")
TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
PII_RE = re.compile(r"PII_[A-Z_]+_\d+")
ATTACH_RE = re.compile(r"\(([a-z]+/[a-zA-Z0-9.+-]+)\)")


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
    """chars/4. There is no usage field in the export - every token count is an estimate."""
    return len(s) // 4


def skills_of(system_text):
    """Skill names from the real <available_skills> block. The tag is also mentioned
    inside <skills_system>, so take the last occurrence."""
    start = system_text.rfind("<available_skills>")
    if start < 0:
        return []
    end = system_text.find("</available_skills>", start)
    block = system_text[start:end] if end > start else system_text[start:]
    return SKILL_RE.findall(block)


def effort_label(items):
    """Observed effort inside the call's history. Counts only MODEL-GENERATED text so
    the label stays orthogonal to prefix size, which is a feature."""
    gen_tokens = tool_calls = assistant_msgs = tool_errors = 0
    for it in items:
        t = it.get("type")
        if t in ("function_call", "custom_tool_call"):
            tool_calls += 1
            payload = it.get("arguments") if t == "function_call" else it.get("input")
            if not isinstance(payload, str):
                payload = json.dumps(payload or "")
            gen_tokens += est_tokens(payload)
        elif it.get("role") == "assistant":
            assistant_msgs += 1
            gen_tokens += est_tokens(_content(it))
        elif t in ("function_call_output", "custom_tool_call_output"):
            head = json.dumps(it.get("output"))[:400].lower()
            if "error" in head or "traceback" in head or "exit code 1" in head:
                tool_errors += 1
    return dict(gen_tokens=gen_tokens, tool_calls=tool_calls,
                assistant_msgs=assistant_msgs, tool_errors=tool_errors)


def extract(req, allow_leaky=False):
    items = req["input"]
    system = next((_content(i) for i in items if i.get("role") == "system"), "")
    first_user = next((i for i in items if i.get("role") == "user"), None)
    fu_parts = []
    if first_user is not None:
        c = first_user.get("content")
        fu_parts = c if isinstance(c, list) else [c]
    user0 = "\n".join(_txt(p) for p in fu_parts)
    tools = req.get("tools") or []
    tool_names = {t.get("name") for t in tools}

    skills = skills_of(system)
    hdr = HDR_RE.search(user0)
    trig = hdr.group(1) if hdr else ""
    day = DAY_RE.search(user0)
    hour = TIME_RE.search(trig) or TIME_RE.search(user0[:4000])

    f = {
        # identity / grouping - never fed to the model
        "trajectory_id": req.get("trajectory_id"),
        "call_index": req.get("call_index"),
        "model": req.get("model"),
        "workspace": hashlib.md5("|".join(sorted(skills)).encode()).hexdigest()[:8],

        # --- source 1: system prompt ---
        "sys_tokens": est_tokens(system),
        "sys_n_skills": len(skills),
        "sys_n_sections": len(set(re.findall(r"<([a-z_]{3,40})>", system))),

        # --- source 2: first user message ---
        "usr_tokens": est_tokens(user0),
        "usr_n_parts": len(fu_parts),
        "usr_has_image": int(any(isinstance(p, dict) and p.get("type") == "input_image"
                                 for p in fu_parts)),
        "usr_has_attachment": int("Attachments:" in user0),
        "usr_n_attachments": len(ATTACH_RE.findall(user0)),
        "usr_audio": int(bool(re.search(r"audio/|\.wav\b|\.mp3\b|\.m4a\b", user0))),
        "usr_n_pii_refs": len(set(PII_RE.findall(user0))),
        "usr_n_urls": user0.count("PII_URL_"),
        "usr_n_questions": user0.count("?"),
        "usr_n_code_fences": user0.count("```"),
        "usr_truncated_ctx": user0.count("[... truncated"),
        "trig_slack": int(trig.startswith("Slack")),
        "trig_teams": int(trig.startswith("Microsoft Teams")),
        "trig_thread_activity": int("New thread activity" in user0),
        "trig_is_dm": int(bool(re.search(r"in #D0[A-Z0-9]+", user0))),
        "trig_weekday": WEEKDAYS.index(day.group(1)) if day else -1,
        "trig_hour": int(hour.group(1)) if hour else -1,

        # --- source 3: tools ---
        "n_tools": len(tools),
        "tools_tokens": est_tokens(json.dumps(tools)),
        "tool_slack": int(any("slack" in (n or "") for n in tool_names)),
        "tool_msteams": int(any("msteams" in (n or "") or "_channel" in (n or "")
                                for n in tool_names)),
        "tool_subagent": int("submit_subagent_result" in tool_names),
        "tool_memory_search": int("memory_search" in tool_names),
        "tool_background": int("wait_for_background_work" in tool_names),
        "tool_view_image": int("view_image" in tool_names),
    }
    for tag in SYS_TAGS:
        f["sys_has_" + tag] = int("<" + tag + ">" in system)
    for tag in USR_TAGS:
        f["usr_has_" + tag] = int("<" + tag in user0)

    if allow_leaky:
        f["LEAK_family_gpt"] = int("apply_patch" in tool_names)
        f["LEAK_has_reasoning"] = int(any(i.get("type") == "reasoning" for i in items))

    f.update(effort_label(items))
    # free text kept for TF-IDF; PII placeholders collapsed so ids are not memorised
    f["_skills_text"] = " ".join(s.strip().replace(" ", "_") for s in skills)
    f["_trigger_text"] = PII_RE.sub("<E>", user0[-6000:])
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir", nargs="?", default="export_linked")
    ap.add_argument("-o", "--out", default="results/features.jsonl")
    ap.add_argument("--allow-leaky", action="store_true",
                    help="include provider-family tells (toolset shape, reasoning items)")
    a = ap.parse_args()

    seen, out = set(), []
    for p in sorted(Path(a.export_dir).glob("trajectories_v1_*.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                req = json.loads(line)
                tid = req.get("trajectory_id")
                if tid in seen:  # one row per task: the earliest sampled call
                    continue
                seen.add(tid)
                out.append(extract(req, a.allow_leaky))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")

    print("wrote {} rows -> {}".format(len(out), a.out))
    print("workspaces={}".format(len(set(r["workspace"] for r in out))))
    print("models={}".format(dict(Counter(r["model"] for r in out))))


if __name__ == "__main__":
    main()
