from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

from src.core import chat, pick_requirement, parse_json_safely, to_rows, write_csv

ROOT = Path(__file__).resolve().parents[2]
REQ_DIR = ROOT / "data" / "requirements"  # directory with .txt requirement files
OUT_DIR = ROOT / "outputs" / "testcase_generated"  # where outputs are written
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "test_cases.csv"  # CSV output path
LAST_RAW_JSON = OUT_DIR / "last_raw.json"  # file where raw LLM text is saved

SYSTEM_PROMPT = """You are a senior QA assistant.
Think step-by-step about the requirement and produce ONLY a JSON array of
test cases using this schema:

[
  {
    "id": "TC-001",
    "title": "Short test title",
    "steps": ["step 1", "step 2"],
    "expected": "Expected result",
    "priority": "High|Medium|Low"
  }
]

Rules:
- Return JSON ONLY (no prose, no fences).
- Provide 5 test cases for a typical requirement.
- Steps should be short, imperative, and precise.
"""

USER_TEMPLATE = 'Requirement:\n"""{requirement_text}"""'

Message = Dict[str, str]

def main() -> None:
    req_path = pick_requirement(sys.argv[1] if len(sys.argv) > 1 else None, REQ_DIR)
    requirement_text = req_path.read_text(encoding="utf-8").strip()
    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(requirement_text=requirement_text),
        },
    ]
    raw = chat(messages)
    try:
        cases = parse_json_safely(raw, LAST_RAW_JSON)
    except Exception as e:
        # gentle retry nudge — a pragmatic teaching technique: show how a
        # small reminder can correct common model format mistakes.
        nudge = (
            raw + "\n\nREMINDER: Return a pure JSON array only, matching the schema."
        )
        try:
            cases = parse_json_safely(nudge, LAST_RAW_JSON)
        except Exception:
            # Surface a clear runtime error with a pointer to the saved raw
            # output so students can debug model responses during the session.
            raise RuntimeError(
                f"Could not parse model output as JSON. See {LAST_RAW_JSON}.\nError: {e}"
            )
    rows = to_rows(cases)
    write_csv(rows, OUT_CSV)
    print(f"✅ Wrote {len(rows)} test cases to: {OUT_CSV.relative_to(ROOT)}")
    print(f"ℹ️  Raw model output saved at: {LAST_RAW_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
