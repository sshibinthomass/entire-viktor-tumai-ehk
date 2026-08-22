#!/usr/bin/env python3
"""Extract deterministic router metrics from one Responses-format request.

No model is called. Semantic fields use explicit regex/structural rules and are
therefore reproducible approximations. The output matches RouteMetrics in
scripts/deterministic_router.py.

Usage:
    python scripts/extract_router_metrics.py request.json --pretty
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from deterministic_router import RouteMetrics


URL_RE = re.compile(r"https?://[^\s)\]>]+", re.I)
WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]+")
POSIX_PATH_RE = re.compile(r"(?<!\w)/(?:[\w.-]+/)+[\w.-]+")
RELATIVE_PATH_RE = re.compile(
    r"\b(?:[\w.-]+[/\\])+[\w.-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|md|txt|csv|sql|html|css|pdf|docx|xlsx|pptx)\b",
    re.I,
)
PLACEHOLDER_RE = re.compile(r"<[A-Z][A-Z0-9_]*>")
CODE_FENCE_RE = re.compile(r"```")
HEADING_RE = re.compile(r"(?m)^\s*#{1,6}\s+")
BULLET_RE = re.compile(r"(?m)^\s*[-*+]\s+")
NUMBERED_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
TABLE_LINE_RE = re.compile(r"(?m)^\s*\|.+\|\s*$")
JSON_LIKE_RE = re.compile(r"[\[{]\s*[\"'][^\"']+[\"']\s*:")
ERROR_RE = re.compile(
    r"\b(error|exception|traceback|failed|failure|invalid|timeout|permission denied|not found|stack trace)\b",
    re.I,
)
ENVIRONMENT_ERROR_RE = re.compile(
    r"\b(network|connection|dns|rate.?limit|quota|permission denied|unauthorized|forbidden|service unavailable|timeout)\b",
    re.I,
)

REQUIREMENT_TERMS = (
    "must",
    "exactly",
    "only",
    "never",
    "do not",
    "don't",
    "preserve",
    "required",
    "without changing",
)

SYSTEM_CLOSE = "</system>"
THREAD_INFO_MARKER = "# === Thread info ==="


def actionable_text(text: str) -> str:
    """Return the smallest deterministic slice that contains the active task.

    Viktor wraps memories, channel instructions, and the live event in one user
    message.  Scoring that entire envelope makes historical words such as
    ``delete`` or ``test`` look like current requirements.  The live Slack event
    is normally after the final ``</system>``.  Scheduled jobs normally keep the
    active task in the final ``Thread info`` block.  Unknown shapes deliberately
    fall back to the full text rather than silently dropping instructions.
    """

    if SYSTEM_CLOSE in text:
        tail = text.rsplit(SYSTEM_CLOSE, 1)[1].strip()
        if tail:
            return tail

    thread_index = text.rfind(THREAD_INFO_MARKER)
    if thread_index >= 0:
        return text[thread_index:].strip()

    return text.strip()


def _matches(text: str, pattern: str) -> bool:
    return re.search(pattern, text, re.I | re.S) is not None


def _has_nonnegated(text: str, pattern: str) -> bool:
    """Match an action unless a nearby phrase explicitly prohibits it."""

    for match in re.finditer(pattern, text, re.I):
        prefix = text[max(0, match.start() - 50) : match.start()]
        if not re.search(
            r"(?:do\s+not|don't|never|must\s+not|should\s+not|no)\b[^.!?\n]{0,35}$",
            prefix,
            re.I,
        ):
            return True
    return False


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    values: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        for key in ("text", "output", "content"):
            value = part.get(key)
            if isinstance(value, str):
                values.append(value)
                break
    return "\n".join(values)


def _item_text(item: dict[str, Any]) -> str:
    values = [_content_text(item.get("content"))]
    for key in ("output", "arguments", "summary"):
        value = item.get(key)
        if isinstance(value, str):
            values.append(value)
    return "\n".join(value for value in values if value)


def _fingerprint(value: Any) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _tool_name(item: dict[str, Any]) -> str:
    return str(item.get("name") or item.get("tool_name") or item.get("type") or "unknown")


def _count_required_arguments(schema: Any) -> int:
    if isinstance(schema, list):
        return sum(_count_required_arguments(value) for value in schema)
    if not isinstance(schema, dict):
        return 0
    count = len(schema.get("required", [])) if isinstance(schema.get("required"), list) else 0
    return count + sum(_count_required_arguments(value) for value in schema.values())


def _primary_intent(text: str) -> str:
    ordered = (
        (
            "debug_prove_or_diagnose",
            r"\b(debug|diagnos(?:e|is|tic)?|root cause|prove|fix\s+(?:the\s+)?(?:bug|error|failure)|why\b.{0,50}\b(?:fail|break))\b",
        ),
        ("plan_or_design", r"\b(plan|design|architect|proposal|strategy|roadmap|blueprint)\b"),
        ("research_or_synthesize", r"\b(research|investigat|look up|find sources|source-backed|synthesi)\b"),
        ("analyze", r"\b(analy[sz]|evaluate|assess|audit|review|reason about|inspect)\b"),
        ("compare", r"\b(compare|contrast|versus|difference between|trade-?off)\b"),
        ("summarize", r"\b(summari[sz]|condense|overview|recap)\b"),
        ("rewrite_or_format", r"\b(rewrite|rephrase|format|translate|polish|proofread)\b"),
        ("extract_or_classify", r"\b(extract|classif|categor|label|list|identify)\b"),
    )
    for intent, pattern in ordered:
        if _matches(text, pattern):
            return intent
    return "extract_or_classify"


def _distinct_matches(patterns: Iterable[re.Pattern[str]], text: str) -> set[str]:
    values: set[str] = set()
    for pattern in patterns:
        values.update(match.group(0) for match in pattern.finditer(text))
    return values


def metrics_from_request(request: dict[str, Any]) -> RouteMetrics:
    input_items = request.get("input", [])
    tools = request.get("tools", [])
    if not isinstance(input_items, list):
        raise ValueError("request.input must be a list")
    if not isinstance(tools, list):
        raise ValueError("request.tools must be a list")

    items = [item for item in input_items if isinstance(item, dict)]
    all_text = "\n".join(_item_text(item) for item in items)
    user_text = "\n".join(
        _item_text(item) for item in items if item.get("role") == "user"
    )
    task_text = actionable_text(user_text) if user_text else actionable_text(all_text)
    lowered = task_text.lower()

    message_counts = {
        role: sum(1 for item in items if item.get("role") == role)
        for role in ("system", "user", "assistant")
    }
    content_parts = [
        part
        for item in items
        for part in (item.get("content") if isinstance(item.get("content"), list) else [])
        if isinstance(part, dict)
    ]
    text_part_count = sum(
        1 for part in content_parts if part.get("type") in {"input_text", "output_text", "text"}
    )
    image_count = sum(1 for part in content_parts if part.get("type") == "input_image")

    function_calls = [item for item in items if item.get("type") == "function_call"]
    custom_calls = [item for item in items if item.get("type") == "custom_tool_call"]
    tool_calls = function_calls + custom_calls
    tool_outputs = [
        item
        for item in items
        if item.get("type") in {"function_call_output", "custom_tool_call_output"}
    ]
    tool_names = [_tool_name(item) for item in tool_calls]
    argument_fingerprints = [_fingerprint(item.get("arguments", "")) for item in tool_calls]
    output_statuses = [
        "error" if ERROR_RE.search(_item_text(item)) else "success" for item in tool_outputs
    ]

    serialized = json.dumps(input_items, ensure_ascii=False, separators=(",", ":"))
    urls = set(URL_RE.findall(task_text))
    paths = _distinct_matches(
        (WINDOWS_PATH_RE, POSIX_PATH_RE, RELATIVE_PATH_RE), task_text
    )
    placeholders = set(PLACEHOLDER_RE.findall(task_text))
    code_fences = len(CODE_FENCE_RE.findall(task_text)) // 2
    headings = len(HEADING_RE.findall(task_text))
    bullets = len(BULLET_RE.findall(task_text))
    numbered = len(NUMBERED_RE.findall(task_text))
    table_lines = len(TABLE_LINE_RE.findall(task_text))
    json_blocks = len(JSON_LIKE_RE.findall(task_text))
    error_markers = len(ERROR_RE.findall(task_text))
    matched_requirement_terms = [term for term in REQUIREMENT_TERMS if term in lowered]

    artifact_terms = set(
        match.group(0).lower()
        for match in re.finditer(
            r"\b(file|document|spreadsheet|workbook|slide|presentation|pdf|repository|repo|database|table|endpoint|service|module|chart|image)\b",
            task_text,
            re.I,
        )
    )
    artifact_count = len(paths) + len(urls) + image_count
    if artifact_count == 0 and artifact_terms:
        artifact_count = len(artifact_terms)
    artifact_count = max(1, artifact_count)

    deliverable_terms = set(
        match.group(0).lower()
        for match in re.finditer(
            r"\b(report|summary|analysis|plan|proposal|implementation|patch|script|chart|table|csv|json|presentation|document|email|message|answer)\b",
            task_text,
            re.I,
        )
    )
    deliverable_count = max(1, len(deliverable_terms))
    dependent_steps = max(numbered, min(bullets, 7))

    has_testing = _matches(
        task_text,
        r"\b(test|tests|testing|verify|verification|validate|lint|typecheck|acceptance criteria|quality check)\b",
    )
    has_write = _has_nonnegated(
        task_text,
        r"\b(edit|modify|patch|implement|refactor|create|delete|remove|rename|move|deploy|publish|send|commit|merge|update)\b",
    )
    has_destructive = _has_nonnegated(
        task_text,
        r"\b(delete|remove|drop|truncate|destroy|wipe|reset|revoke|overwrite|force push)\b",
    )
    has_external = _has_nonnegated(
        task_text,
        r"\b(deploy|publish|send|post|email|message|merge|push|production|public)\b",
    )
    has_search = _has_nonnegated(
        task_text,
        r"\b(search|browse|look up|research|find sources|latest|current|online|web)\b",
    )
    has_read_artifact = artifact_count > 1 or bool(paths or urls)
    multi_file = (
        len(paths) > 1
        or _matches(task_text, r"\b(multiple|several|all|across) (files|modules|documents|services)\b")
    ) and has_write

    if has_write:
        expected_action = (
            "multiple_writes_or_chained_tools" if multi_file or has_external or has_testing else "one_local_write"
        )
    elif has_search:
        expected_action = "search_or_multiple_reads"
    elif has_read_artifact:
        expected_action = "one_read_only"
    else:
        expected_action = "no_tool"

    stages = 0
    for flag in (has_search, has_read_artifact, has_write, has_testing, has_external):
        stages += int(flag)
    if expected_action != "no_tool":
        stages = max(1, stages)

    high_stakes = _matches(
        task_text,
        r"\b(medical|medicine|diagnosis|patient|legal|lawsuit|contract|financial advice|investment|safety-critical|emergency)\b",
    )
    security = _matches(
        task_text,
        r"\b(security|secret|password|credential|authentication|authorization|permission|access control|token|api key|vulnerabilit)\b",
    )
    if has_destructive or _matches(task_text, r"\b(permission-changing|production deployment)\b"):
        action_risk = "destructive_public_deployment_or_permission_change"
    elif has_external:
        action_risk = "persistent_or_external"
    elif has_write:
        action_risk = "reversible_local_write"
    else:
        action_risk = "read_only_or_none"

    strict_schema = _matches(
        task_text, r"\b(json schema|strict schema|machine-readable|valid json|csv columns|exact format)\b"
    )
    citations = _matches(
        task_text,
        r"\b(cite|citation|provide sources|include sources|sources required|references required|source traceability)\b",
    )
    preservation = _matches(
        task_text,
        r"\b(preserve|backward compatible|compatibility|existing behavior|same style|template|do not change)\b",
    )

    repeated_failed = 0
    seen_failed: set[tuple[str, str]] = set()
    for call, fingerprint in zip(tool_calls, argument_fingerprints):
        key = (_tool_name(call), fingerprint)
        if key in seen_failed:
            repeated_failed += 1
        seen_failed.add(key)

    environmental_errors = sum(
        1 for item in tool_outputs if ENVIRONMENT_ERROR_RE.search(_item_text(item))
    )

    return RouteMetrics(
        request_model=request.get("model"),
        serialized_character_count=len(serialized),
        estimated_input_tokens=len(serialized) // 4,
        input_item_count=len(items),
        system_message_count=message_counts["system"],
        user_message_count=message_counts["user"],
        assistant_message_count=message_counts["assistant"],
        text_part_count=text_part_count,
        reasoning_item_count=sum(1 for item in items if item.get("type") == "reasoning"),
        input_image_count=image_count,
        function_tool_call_count=len(function_calls),
        custom_tool_call_count=len(custom_calls),
        tool_output_count=len(tool_outputs),
        unique_tool_count=len(set(tool_names)),
        tool_call_sequence=tool_names,
        tool_argument_fingerprints=argument_fingerprints,
        tool_output_statuses=output_statuses,
        tool_definition_count=len(tools),
        tool_schema_character_count=len(json.dumps(tools, ensure_ascii=False)),
        required_tool_argument_count=_count_required_arguments(tools),
        code_fence_count=code_fences,
        url_count=len(urls),
        file_path_count=len(paths),
        placeholder_entity_count=len(placeholders),
        heading_count=headings,
        bullet_count=bullets,
        numbered_step_count=numbered,
        table_count=table_lines // 2,
        json_like_block_count=json_blocks,
        error_or_stacktrace_marker_count=error_markers,
        explicit_requirement_term_count=len(matched_requirement_terms),
        matched_requirement_terms=matched_requirement_terms,
        distinct_referenced_artifact_count=artifact_count,
        distinct_input_source_count=max(1, int(bool(task_text)) + len(urls) + image_count),
        primary_intent=_primary_intent(task_text),
        dependent_step_count=dependent_steps,
        has_cross_reference_or_synthesis=_matches(
            task_text, r"\b(across|cross-reference|combine|synthesi|reconcile|using all|between .* and)\b"
        ),
        has_tradeoffs_or_competing_objectives=_matches(
            task_text, r"\b(trade-?off|balance|pros and cons|versus|while minimizing|subject to)\b"
        ),
        has_counterfactual_uncertainty_or_scenarios=_matches(
            task_text, r"\b(what if|counterfactual|uncertain|uncertainty|scenario|sensitivity|confidence interval)\b"
        ),
        has_contradiction_or_ambiguity_to_resolve=_matches(
            task_text, r"\b(contradict|conflict|ambiguous|ambiguity|inconsistent|unclear)\b"
        ),
        artifact_object_count=artifact_count,
        deliverable_count=deliverable_count,
        has_cross_file_or_system_dependency=multi_file or _matches(
            task_text, r"\b(cross-system|integration|between services|across modules|dependency)\b"
        ),
        has_ordered_dependent_workflow=dependent_steps >= 2 or stages >= 3,
        has_stateful_coordination=_matches(
            task_text, r"\b(workflow|stateful|transaction|migration|orchestrat|coordinate|multi-agent)\b"
        ),
        is_multi_file_modification=multi_file,
        expected_action=expected_action,
        expected_tool_stage_count=stages,
        requires_testing_or_verification=has_testing,
        requires_tool_chaining=stages >= 2,
        has_code_sql_formula_or_data_transformation=(
            code_fences > 0
            or _matches(task_text, r"\b(code|python|javascript|typescript|sql|formula|query|dataframe|algorithm)\b")
        ),
        has_debug_error_failing_test_or_stacktrace=(
            error_markers > 0
            or _matches(task_text, r"\b(debug|failing test|stack trace|root cause)\b")
        ),
        has_architecture_migration_concurrency_or_integration=_matches(
            task_text, r"\b(architecture|migration|concurren|integration|distributed|race condition|system design)\b"
        ),
        has_formal_math_algorithms_security_or_specialist_domain=(
            security
            or _matches(task_text, r"\b(theorem|proof|formal math|optimization|cryptograph|specialist|domain-specific)\b")
        ),
        has_cross_domain_reasoning=_matches(
            task_text, r"\b(cross-domain|interdisciplinary|technical and business|legal and technical)\b"
        ),
        hard_constraint_count=len(matched_requirement_terms),
        requires_strict_schema_or_machine_output=strict_schema,
        requires_citations_or_source_traceability=citations,
        requires_compatibility_style_template_or_behavior_preservation=preservation,
        requires_mutually_consistent_outputs=deliverable_count > 1 and _matches(
            task_text, r"\b(consistent|match|aligned|same values|correspond)\b"
        ),
        has_exact_numerical_or_factual_acceptance_criteria=_matches(
            task_text, r"\b(exact|precise|acceptance criteria|must equal|no errors|100%)\b"
        ),
        action_risk=action_risk,
        is_high_stakes_domain=high_stakes,
        involves_security_secrets_authentication_or_permissions=security,
        has_irreversibility_or_broad_blast_radius=has_destructive or _matches(
            task_text, r"\b(irreversible|cannot be undone|all users|entire database|broad blast radius)\b"
        ),
        requires_ocr_spatial_or_chart_reasoning=_matches(
            task_text, r"\b(ocr|read the image|spatial|chart|graph|diagram|screenshot)\b"
        ),
        requires_layout_sensitive_artifact=_matches(
            task_text, r"\b(layout|pixel-perfect|presentation|slides?|pdf|docx|spreadsheet|xlsx|formatting)\b"
        ),
        malformed_tool_or_structured_response_count=0,
        repeated_failed_tool_and_arguments_count=repeated_failed,
        explicit_user_correction_count=sum(
            1
            for item in items
            if item.get("role") == "user"
            and _matches(_item_text(item), r"\b(no,|that's wrong|that is wrong|incorrect|try again|you missed)\b")
        ),
        consecutive_model_attributable_failure_count=0,
        failed_verification_after_claimed_completion_count=sum(
            1
            for item in tool_outputs
            if _matches(_item_text(item), r"\b(test|verification|validation).{0,40}\b(fail|error)\b")
        ),
        no_progress_two_call_window_count=0,
        environmental_error_count=environmental_errors,
        shared_prefix_tokens=0,
        estimated_uncached_tokens_after_switch=len(serialized) // 4,
        switch_count=0,
    )


def metrics_dict_from_request(request: dict[str, Any]) -> dict[str, Any]:
    return asdict(metrics_from_request(request))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON file containing one request")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(json.dumps(metrics_dict_from_request(request), indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
