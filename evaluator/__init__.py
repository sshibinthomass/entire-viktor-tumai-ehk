"""Evaluator: rule-based difficulty (1/2/3) from the FULL trajectory.

Unlike the router, the evaluator is allowed to see everything the export
holds for a task: every tool call, assistant turn, reasoning item, tool
error and the final context size. It grades how hard the task actually was
in hindsight, giving an independent yardstick to judge the router against —
without ever using the logged model id as ground truth.
"""
