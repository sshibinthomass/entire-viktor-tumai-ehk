"""Router: assigns Tier 1/2/3 to a task from routing-time information only.

Inputs per task (nothing else):
  1. the first user message (the trigger)
  2. the system prompt (the agent harness)
  3. the tool definitions available to the call

The logged `model` field is NEVER read here — it is not ground truth for
difficulty (the historical dispatch policy is unknown and biased).
"""
