#!/usr/bin/env python3
"""DEPRECATED — sequential ids are now assigned by enrich_dataset.py.

The old version of this script rewrote export/*.jsonl IN PLACE (truncation risk
on crash, silently dropped unknown fields, broke the documented 3-field raw
schema). The raw export must stay untouched: it is the licensed source data.

Everything this script did now lives in scripts/enrich_dataset.py, which reads
the raw export and writes enriched copies (request_id / trajectory_id /
call_index) to export_linked/ via write-to-temp-then-rename.

Usage:
    python scripts/enrich_dataset.py export/ export_linked/
"""
import sys

sys.exit("number_dataset.py is deprecated: it mutated the raw export in place.\n"
         "Run  python scripts/enrich_dataset.py export/ export_linked/  instead —\n"
         "it assigns the same sequential ids without touching export/*.jsonl.")
