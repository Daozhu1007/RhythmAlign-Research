"""Structural QA for S0 deliverables; does not claim scientific reproduction."""
from pathlib import Path
import hashlib
import json
import re

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
required = [
    "S0_RESEARCH_STUDY.md", "LITERATURE_MAP.md", "ARCHITECTURE_OPTIONS.md",
    "DATA_AND_GROUND_TRUTH_PLAN.md", "MINIMAL_DECISIVE_EXPERIMENT.md",
    "PAPER_DIRECTION_MEMO.md", "S0_DECISION.json", "EVIDENCE_AND_SCOPE.md",
    "WORKSPACE_EVIDENCE_CHECKS.json",
]
errors = []
for name in required:
    if not (OUT / name).is_file():
        errors.append(f"Missing deliverable: {name}")

main = (OUT / "S0_RESEARCH_STUDY.md").read_text(encoding="utf-8")
parts = [int(x) for x in re.findall(r"^## Part (\d+) —", main, re.M)]
if parts != list(range(1, 19)):
    errors.append(f"Main study parts incorrect: {parts}")
expected_end = [
    "## 1. Feasibility verdict", "## 2. Best next technical route",
    "## 3. Need for new model", "## 4. Ground-truth priority",
    "## 5. Minimal decisive experiment", "## 6. Paper outlook",
    "## 7. Top five actions",
]
headings = re.findall(r"^## .+$", main, re.M)
if headings[-7:] != expected_end:
    errors.append("Required final seven headings do not match")
last = main.split(expected_end[-1], 1)[-1]
if len(re.findall(r"^[1-5]\. ", last, re.M)) != 5:
    errors.append("Final section must contain five ranked actions")

decision = json.loads((OUT / "S0_DECISION.json").read_text(encoding="utf-8"))
for key, expected in [("feasibility", "A"), ("need_for_new_model", "B"), ("paper_outlook", "B")]:
    if decision[key]["choice"] != expected:
        errors.append(f"Decision choice inconsistent: {key}")
if decision["best_next_route"]["id"] != "A":
    errors.append("Next route must be A")

local_links = set()
web_links = set()
documents = {}
for path in sorted(OUT.glob("*.md")):
    content = path.read_text(encoding="utf-8")
    documents[path.name] = {"bytes": path.stat().st_size, "words_approx": len(content.split())}
    if "\ufffd" in content:
        errors.append(f"Replacement character in {path.name}")
    for target in re.findall(r"\]\(([^)]+)\)", content):
        target = target.strip("<>")
        if re.match(r"^[A-Za-z]:[/\\]", target):
            local_links.add(target)
            if not Path(target).exists():
                errors.append(f"Broken local link in {path.name}: {target}")
        elif target.startswith(("https://", "http://")):
            web_links.add(target)
    if re.search(r"turn\d+(?:search|view|fetch)\d+", content):
        errors.append(f"Internal web citation identifier in {path.name}")
    table_width = None
    for line_number, line in enumerate(content.splitlines(), 1):
        if line.startswith("|"):
            width = len(re.findall(r"(?<!\\)\|", line))
            if table_width is None:
                table_width = width
            elif width != table_width:
                errors.append(f"Table width mismatch in {path.name}:{line_number}")
        else:
            table_width = None

receipt = json.loads((OUT / "WORKSPACE_EVIDENCE_CHECKS.json").read_text(encoding="utf-8"))
hash_checks = {}
for rel, expected in receipt["sha256"].items():
    observed = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    hash_checks[rel] = observed == expected
    if observed != expected:
        errors.append(f"Historical evidence hash changed: {rel}")

report = {
    "status": "pass" if not errors else "fail",
    "scope": "Required files, 18 study parts, final seven sections, JSON choices, local links, encoding and historical hashes; web content was researched separately",
    "required_files": required,
    "documents": documents,
    "unique_local_links_checked": len(local_links),
    "unique_primary_web_urls_in_deliverables": len(web_links),
    "historical_hashes_unchanged": hash_checks,
    "errors": errors,
}
(OUT / "DELIVERABLE_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": report["status"], "documents": documents,
                  "local_links": len(local_links), "web_urls": len(web_links),
                  "historical_hashes": len(hash_checks), "errors": errors}, indent=2))
raise SystemExit(bool(errors))
