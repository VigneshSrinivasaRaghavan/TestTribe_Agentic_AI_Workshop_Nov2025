from __future__ import annotations

import sys
import logging
import re
from pathlib import Path
from typing import List, Dict
from src.core import chat, pick_requirement, parse_json_safely, to_rows, write_csv
from langchain_core.prompts import PromptTemplate
from src.integrations.testrail import map_case_to_testrail_payload, create_case, list_cases
from src.integrations.testrail import add_result

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

def _norm(title: str | None) -> str:
    """
    Normalize a title for stable dedupe.
    - case-insensitive
    - trims
    - removes non-alphanumeric (keeps [a-z0-9] only)
    - collapses whitespace
    """
    s = (title or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


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
    
    # --- Day-4: Act step → push to TestRail mock ---
    logger.info("ℹ️  Starting TestRail push step")

    # Map once → collect payloads (so we dedupe on the exact titles we will POST)
    payloads: list[dict] = []
    for idx, c in enumerate(cases, start=1):
        try:
            p = map_case_to_testrail_payload(c)
            payloads.append(p)
        except Exception as e:
            logger.warning("Skipping case %s (mapping error): %s", c.get("id") or idx, e)

    # Build once: incoming titles from *mapped* payloads
    incoming_titles = { _norm(p.get("title")) for p in payloads }

    # Build once: existing titles from TestRail (project-wide)
    try:
        existing = list_cases()  # returns list[dict]
        existing_titles = { _norm(case.get("title")) for case in existing }
    except Exception as e:
        logger.warning("Could not fetch existing titles; proceeding without dedupe: %s", e)
        existing_titles = set()

    logger.info("📚 Loaded %d existing titles from TestRail (project-wide)", len(existing_titles))

    # One-shot duplicate report (informational)
    dupes = incoming_titles & existing_titles
    if dupes:
        logger.info(
            "🚧 Detected %d duplicate title(s) in this batch; they will be skipped: %s",
            len(dupes), sorted(list(dupes))[:5]  # show first few only
        )
    else:
        logger.info("✅ No duplicates detected for this batch")

    created_ids: list[int] = []
    for p in payloads:
        title_norm = _norm(p.get("title"))

        # Skip if already exists (pre-existing or created earlier in this run)
        if title_norm in existing_titles:
            logger.info("↪️  Skipping existing case: %s", p.get("title"))
            continue

        try:
            res = create_case(p)
            cid = res.get("id")
            if cid is not None:
                created_ids.append(int(cid))         # ✅ safe append
                existing_titles.add(title_norm)      # prevent same-batch duplicates
                # Seed an initial result for visibility (Untested = 3)
                try:
                    _ = add_result(int(cid), status_id=3, comment="Seeded by agent on create")
                except Exception as e:
                    logger.warning("Could not seed result for case %s: %s", cid, e)
            else:
                logger.warning("Create case response missing 'id': %s", res)
        except Exception as e:
            logger.error("Create case failed for '%s': %s", p.get("title"), e)

    logger.info("📌 Created %d TestRail cases: %s", len(created_ids), created_ids)

    # Quick verification
    try:
        all_cases = list_cases()
        logger.info("🧾 TestRail now has %d cases in project", len(all_cases))
    except Exception as e:
        logger.warning("Could not list TestRail cases: %s", e)

    logger.info("✅ Test cases pushed to TestRail successfully with id %s", created_ids)


    
    logger.info("✅ Test cases pushed to TestRail successfully with id %s", created_ids)
    logger.info("✅ Wrote %d test cases to: %s", len(rows), OUT_CSV.relative_to(ROOT))  
    logger.info("ℹ️  Raw model output saved at: %s", LAST_RAW_JSON.relative_to(ROOT))

if __name__ == "__main__":
    main()
