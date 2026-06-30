"""
Run this against queries.jsonl before committing any staleness items.
Flags text that contains recency words or explicit date stamps that make
V1 trivially distinguishable from V2 — the anti-cheat checklist from the spec.
"""
import json
import re
import sys

RECENCY_WORDS = [
    "former", "formerly", "previous", "previously", "prior", "prior to",
    "outdated", "no longer", "used to", "once was", "at the time",
    "replaced", "replacing", "superseded", "deprecated",
    "stepped down", "took over", "succeeded", "departed", "departure",
    "transition", "transitioning",
]

DATE_PATTERN = re.compile(
    r"\b(19|20)\d{2}\b|"          # 4-digit year (1900s-2000s)
    r"\b(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\s+\d{4}\b|"
    r"\bq[1-4]\s+\d{4}\b",        # e.g. Q3 2024
    re.IGNORECASE,
)


def check_item(item: dict) -> list[str]:
    issues = []
    if item["type"] not in ("staleness_sensitive", "historical_belief"):
        return issues

    for field in ("v1_text", "v2_text"):
        text = item.get(field, "") or ""
        text_lower = text.lower()

        for word in RECENCY_WORDS:
            if word in text_lower:
                issues.append(f"[{item['id']}] {field} contains recency word: '{word}'")

        for match in DATE_PATTERN.finditer(text):
            issues.append(f"[{item['id']}] {field} contains date stamp: '{match.group()}'")

    return issues


def main(path="data/queries.jsonl"):
    with open(path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    staleness_items = [i for i in items if i["type"] in ("staleness_sensitive", "historical_belief")]
    print(f"Checking {len(staleness_items)} staleness/historical items...\n")

    all_issues = []
    for item in staleness_items:
        all_issues.extend(check_item(item))

    if all_issues:
        print("FAILED - issues found:")
        for issue in all_issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("PASSED - no recency words or date stamps found.")


if __name__ == "__main__":
    main()
