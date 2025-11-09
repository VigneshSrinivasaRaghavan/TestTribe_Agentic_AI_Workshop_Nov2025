# Day 1 — Edge / Negative Testcase Generator

Question

Implement an agent `edgecase_agent` that reads a requirement text (single file) and produces a JSON array of test cases emphasizing edge and negative scenarios in addition to normal cases.

Requirements

- Input: a requirement text file from `data/requirements` (CLI: `--input PATH`).
- Output: a JSON array (printed or returned) and a CSV `outputs/testcase_generated/test_cases.csv`.
- Each test case object must include: `id`, `title`, `steps`, `expected`, `priority`, `tags`, `likelihood`.
- At least 6 and at most 12 test cases. At least 30% must be tagged as `edge` or `negative`.

Expected output (JSON schema per case)

{
  "id": "TC-001",
  "title": "Short test title",
  "steps": ["step 1", "step 2"],
  "expected": "Expected result",
  "priority": "High|Medium|Low",
  "tags": ["edge"|"negative"|"happy"],
  "likelihood": "High|Medium|Low"
}

CSV columns: TestID, Title, Steps, Expected, Priority, Tags, Likelihood

