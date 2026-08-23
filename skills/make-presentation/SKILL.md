---
name: make-presentation
description: Build a Viktor-branded presentation of the team's solution, structured for the 5-minute judged defense.
---

# /make-presentation — the 5-minute defense deck

Judges score presentation clarity as: **one chart, one claim, one known weakness — in 5 minutes.**

## 1. Collect from the team (ask, don't invent)
- Team name + members
- The objective they chose (cost / latency / quality / own trade-off) and why
- The routing signal (what structure in the traces does the router use — must be nameable)
- Headline numbers from the **held-out split**, priced cache-aware
- The off-policy method (matching / weighting / judge model) and its weakest failure mode

## 2. Build from the template
- Copy `templates/presentation.html` and fill the marked `<!-- FILL -->` slots. It is
  self-contained (no network needed), arrow keys / space to navigate, `f` for fullscreen.
- Slide order is fixed on purpose, and each slide owns a slot in the 5 minutes:
  1. **Title + the claim** (0:00–0:20) — one line, the number, no agenda slide
  2. **The setup in 3 facts** (0:20–1:00) — what the log is, what is observable, what is not
  3. **The frontier** (1:00–2:00) — the one chart: router as a curve, four baselines as
     labeled dots (always-cheap, always-strong, logged policy, random at matched cost),
     error bars, and say out loud where you would operate
  4. **The router** (2:00–2:45) — the real decision rule: thresholds, features, fallback
  5. **Honesty** (2:45–3:45) — observed vs extrapolated, the calibration number, cache-reset
     cost on switches, and the two things offline eval cannot see
  6. **One known weakness + a week** (3:45–4:20) — volunteered and specific, fix sketched
  7. **Close** (4:20–5:00) — restate the opening claim with the number, one line on generality
  8. **Appendix** — rehearsed Q&A openers (why not always-cheapest? what breaks at 2x
     traffic? what would change your mind?)
- Keep slides text-minimal: title as a sentence, one idea per slide, numbers always with a
  denominator and an interval. Put the prose in speaker notes, not on the wall.
- Brand: violet `#6748FD` on navy `#150079`, peach `#FFBD9E` accents — already in the template.
  Don't add extra colors; don't add hype words. Short, active sentences.

## 3. Quality gate before you say "done"
- Every number on a slide traces to a script output the team can rerun.
- The weakness slide names a real failure mode (cache reset, selection bias, judge-model
  disagreement…) — an honest weakness scores; a hidden one costs.
- Open the file in a browser and step through all slides; check nothing overflows.
