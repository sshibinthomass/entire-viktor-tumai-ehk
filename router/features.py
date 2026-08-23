"""Routing-time feature extraction.

Feature sources are restricted to what a router knows BEFORE the agent runs:
  1. the system prompt
  2. the FIRST user message (Viktor's auto-read envelope + the trigger)
  3. the `tools` definitions

Everything downstream of dispatch (assistant turns, tool calls, outputs,
reasoning items, the logged `model`) is off-limits — that belongs to the
evaluator, which judges the router from the full trajectory.

Feature names follow results/features.jsonl from the earlier exploration so
the two artifacts stay comparable.
"""
import json
import re

SYS_TAGS = ["available_skills", "slack_history", "teams_history", "structured_output",
            "voice", "personalization", "custom_instructions", "personality",
            "thread_instructions", "core_philosophy", "operating_rules", "work_approach"]

USR_TAGS = ["auto_read_learnings", "auto_read_execution_log", "auto_read_channel_instructions",
            "auto_read_personal_skill", "auto_read_recent_activity", "delivery_note",
            "summary_so_far"]

SKILL_RE = re.compile(r"^\-\s\*\*([a-zA-Z0-9_\- ]+)\*\*", re.M)
HDR_RE = re.compile(r"\[((?:Slack|Microsoft Teams|Email|Cron|Scheduled|New)[^\]\n]{0,160})\]")
PII_RE = re.compile(r"PII_[A-Z_]+_\d+")
ATTACH_RE = re.compile(r"\(([a-z]+/[a-zA-Z0-9.+-]+)\)")

# imperative verbs that usually open a multi-step build/analysis ask
ACTION_RE = re.compile(r"\b(create|build|write|generate|analy[sz]e|summari[sz]e|research|"
                       r"prepare|draft|review|update|fix|implement|compare|plan|schedule|"
                       r"organize|extract|convert|translate|deploy)\b", re.I)
COORD_RE = re.compile(r"\b(and then|after that|once done|as well as|also|finally|steps?:)\b", re.I)


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
    """chars/4 — the export has no usage field, every token count is an estimate."""
    return len(s) // 4


def skills_of(system_text):
    """Skill names from the real <available_skills> block (last occurrence — the
    tag is also mentioned inside <skills_system>)."""
    start = system_text.rfind("<available_skills>")
    if start < 0:
        return []
    end = system_text.find("</available_skills>", start)
    block = system_text[start:end] if end > start else system_text[start:]
    return SKILL_RE.findall(block)


def extract(req):
    """Routing-time features for one request (use the trajectory's EARLIEST call)."""
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

    f = {
        # identity only — never a feature
        "trajectory_id": req.get("trajectory_id"),

        # --- source 1: system prompt (harness size & configuration) ---
        "sys_tokens": est_tokens(system),
        "sys_n_skills": len(skills),
        "sys_n_sections": len(set(re.findall(r"<([a-z_]{3,40})>", system))),
        "sys_n_flags": sum(int("<" + t + ">" in system) for t in SYS_TAGS),

        # --- source 2: first user message (the ask) ---
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
        "usr_n_action_verbs": len(ACTION_RE.findall(user0)),
        "usr_n_coord_markers": len(COORD_RE.findall(user0)),
        "usr_truncated_ctx": user0.count("[... truncated"),
        "usr_n_ctx_blocks": sum(int("<" + t in user0) for t in USR_TAGS),
        "trig_slack": int(trig.startswith("Slack")),
        "trig_teams": int(trig.startswith("Microsoft Teams")),
        "trig_scheduled": int(trig.startswith(("Cron", "Scheduled"))),
        "trig_thread_activity": int("New thread activity" in user0),

        # --- source 3: tools (capability surface offered to the agent) ---
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

    # short PII-collapsed preview for the dashboard table (local use only)
    body = user0
    if hdr:
        body = user0[hdr.end():]
    f["_preview"] = PII_RE.sub("<E>", " ".join(body.split()))[:140]
    f["_trigger"] = PII_RE.sub("<E>", trig)[:60]
    return f


# Features the unsupervised router scores on, grouped by what they measure.
# Signs are part of the routing insight (measured against observed effort):
#   ask       (+) a dense ask — many entities, questions, action verbs — is hard
#   harness   (+) a heavily configured harness marks a complex workspace
#   breadth   (-) broad generic toolsets go with quick conversational turns;
#                 focused subagent toolsets do the heavy lifting
#   midthread (-) mid-thread triggers (truncated ctx, auto-read blocks) mean
#                 most of the work already happened
FEATURE_GROUPS = {
    "ask": {"sign": 1, "features": [
        "usr_n_pii_refs", "usr_n_questions", "usr_n_action_verbs",
        "usr_n_coord_markers", "usr_n_urls", "usr_tokens", "usr_n_attachments",
        "usr_audio", "usr_has_image", "usr_n_code_fences"]},
    "harness": {"sign": 1, "features": [
        "sys_n_sections", "sys_n_flags", "sys_n_skills", "sys_tokens",
        "tool_subagent"]},
    "breadth": {"sign": -1, "features": [
        "n_tools", "tools_tokens", "tool_background", "tool_memory_search"]},
    "midthread": {"sign": -1, "features": [
        "usr_truncated_ctx", "usr_n_ctx_blocks", "usr_n_parts",
        "trig_thread_activity"]},
}
