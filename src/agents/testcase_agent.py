from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict

from src.core import chat, pick_requirement, parse_json_safely, to_rows, write_csv
from langchain_core.prompts import PromptTemplate
import logging

ROOT = Path(__file__).resolve().parents[2]
REQ_DIR = ROOT / "data" / "requirements"  # directory with .txt requirement files
OUT_DIR = ROOT / "outputs" / "testcase_generated"  # where outputs are written
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "test_cases.csv"  # CSV output path
LAST_RAW_JSON = OUT_DIR / "last_raw.json"  # file where raw LLM text is saved
PROMPTS_DIR = ROOT / "src" / "core" / "prompts"
SYSTEM_PROMPT = (PROMPTS_DIR/ "testcase_system.txt").read_text(encoding="utf-8")
USER_TEMPLATE = (PROMPTS_DIR/ "testcase_user.txt").read_text(encoding="utf-8")

Message = Dict[str, str]

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    req_path = pick_requirement(sys.argv[1] if len(sys.argv) > 1 else None, REQ_DIR)
    requirement_text = req_path.read_text(encoding="utf-8").strip()
    messages: List[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user","content": USER_TEMPLATE.format(requirement_text=requirement_text)},
    ]
    logger.info("Calling chat: provider payload msgs=%d (sys=1,user=1)", len(messages))
    raw = chat(messages)
    try:
        cases = parse_json_safely(raw, LAST_RAW_JSON)
    except Exception as e:
        # gentle retry nudge — a pragmatic teaching technique: show how a
        # small reminder can correct common model format mistakes.
        logger.exception("Initial parse_json_safely failed; will nudge and retry. Raw saved at %s",LAST_RAW_JSON,)
        nudge = (
            raw + "\n\nREMINDER: Return a pure JSON array only, matching the schema."
        )
        try:
            cases = parse_json_safely(nudge, LAST_RAW_JSON)
        except Exception:
            # Surface a clear runtime error with a pointer to the saved raw
            # output so students can debug model responses during the session.
            logger.error("Could not parse model output after retry; see %s", LAST_RAW_JSON
)
            raise RuntimeError(
                f"Could not parse model output as JSON. See {LAST_RAW_JSON}.\nError: {e}"
            )
    rows = to_rows(cases)
    write_csv(rows, OUT_CSV)
    logger.info("✅ Wrote %d test cases to: %s", len(rows), OUT_CSV.relative_to(ROOT))
    logger.info("ℹ️  Raw model output saved at: %s", LAST_RAW_JSON.relative_to(ROOT))

if __name__ == "__main__":
    main()
