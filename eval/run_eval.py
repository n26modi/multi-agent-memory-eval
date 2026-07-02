"""
Eval harness — runs both memory conditions against the 30-query dataset.

Usage:
    # run Chroma condition (no Neo4j needed)
    python eval/run_eval.py --backend chroma

    # run Graphiti condition (requires Neo4j running)
    python eval/run_eval.py --backend graphiti

    # cap daily queries to stay within Groq free-tier token limits
    python eval/run_eval.py --backend chroma --cap 10

    # print results table from completed runs
    python eval/run_eval.py --report
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from graph import build_graph
from memory.base import Finding
from memory.chroma_memory import ChromaMemory
from state import ResearchState

QUERIES_PATH = Path("data/queries.jsonl")
RESULTS_DIR = Path("eval/results")
RESULTS_DIR.mkdir(exist_ok=True)


# ── query loading ────────────────────────────────────────────────────────────

def load_queries() -> list[dict]:
    with open(QUERIES_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


# ── memory seeding ───────────────────────────────────────────────────────────

async def seed_memory(q: dict, memory) -> None:
    """Write corpus documents into memory before running the query."""
    if q["type"] == "static_fact":
        await memory.write(Finding(
            entity=q["entity"],
            relation=q["relation"],
            value=q["ground_truth"],
            source_id=q["id"],
            confidence=1.0,
            valid_at=time.time(),
            retrieved_text=q["text"],
        ))
    else:
        # staleness_sensitive and historical_belief both get V1 then V2
        # V1 written first (older timestamp), V2 second (newer) so Graphiti
        # correctly marks V1 as superseded when V2 arrives
        await memory.write(Finding(
            entity=q["entity"],
            relation=q["relation"],
            value=q["v1_value"],
            source_id=f"{q['id']}_v1",
            confidence=1.0,
            valid_at=q["v1_valid_at"],
            retrieved_text=q["v1_text"],
        ))
        await memory.write(Finding(
            entity=q["entity"],
            relation=q["relation"],
            value=q["ground_truth"] if q["type"] == "staleness_sensitive" else q["v1_value"],
            source_id=f"{q['id']}_v2",
            confidence=1.0,
            valid_at=q["v2_valid_at"],
            retrieved_text=q["v2_text"],
        ))


# ── scoring ──────────────────────────────────────────────────────────────────

def score(result: ResearchState, q: dict) -> dict:
    report = result["final_report"].lower()
    gt = q["ground_truth"].lower()
    v1 = (q.get("v1_value") or "").lower()
    tags = result["failure_tags"]

    correct = gt in report
    used_stale = bool(v1) and v1 in report and not correct
    # false-approve: stale fact reached the final report without being caught
    critic_false_approve = used_stale and result["final_report"] != "escalated"
    # staleness caused failure: either false-approve OR escalated because critic
    # kept catching V1 but the researcher couldn't surface V2
    staleness_caused_failure = not correct and "staleness_failure" in tags

    return {
        "qid": q["id"],
        "type": q["type"],
        "correct": correct,
        "used_stale_fact": used_stale,
        "critic_false_approve": critic_false_approve,
        "staleness_caused_failure": staleness_caused_failure,
        "retries": result["retry_count"],
        "exit": "escalated" if result["final_report"] == "escalated" else "approved",
        "failure_tags": tags,
    }


# ── persistence ──────────────────────────────────────────────────────────────

def results_path(backend: str) -> Path:
    return RESULTS_DIR / f"loop_state_{backend}.json"


def load_completed(backend: str) -> dict:
    p = results_path(backend)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f).get("queries", {})


def persist(backend: str, qid: str, row: dict) -> None:
    p = results_path(backend)
    state = {"backend": backend, "queries": load_completed(backend)}
    state["queries"][qid] = row
    with open(p, "w") as f:
        json.dump(state, f, indent=2)


# ── main eval loop ───────────────────────────────────────────────────────────

async def run_condition(backend: str, queries: list[dict], cap: int | None = None) -> list[dict]:
    completed = load_completed(backend)
    rows, done_today = [], 0

    for q in queries:
        if q["id"] in completed:
            rows.append(completed[q["id"]])
            print(f"  [{q['id']}] skipped (already done)")
            continue

        if cap and done_today >= cap:
            print(f"\nHit daily cap ({cap}). Resume tomorrow with the same command.")
            break

        print(f"  [{q['id']}] {q['type']} — {q['query'][:60]}")

        # fresh memory per query for isolation
        if backend == "chroma":
            memory = ChromaMemory(collection_name=q["id"])
        elif backend == "graphiti":
            from memory.graphiti_memory import GraphitiMemory
            memory = GraphitiMemory(
                uri=os.environ["NEO4J_URI"],
                user=os.environ["NEO4J_USER"],
                password=os.environ["NEO4J_PASSWORD"],
                group_id=q["id"],
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        await seed_memory(q, memory)

        graph = build_graph(memory)
        initial: ResearchState = {
            "query": q["query"],
            "subtasks": [],
            "findings": [],
            "approved": [],
            "retry_count": 0,
            "max_retries": 2,
            "last_quality": 0.0,
            "failure_tags": [],
            "final_report": "",
            "temporal_context": q["v1_valid_at"] if q["type"] == "historical_belief" else None,
        }

        t0 = time.time()
        result = await graph.ainvoke(initial)
        latency_ms = int((time.time() - t0) * 1000)

        row = {**score(result, q), "latency_ms": latency_ms}
        rows.append(row)
        persist(backend, q["id"], row)
        done_today += 1

        status = "✓" if row["correct"] else "✗"
        print(f"         {status} correct={row['correct']} stale={row['used_stale_fact']} latency={latency_ms}ms")

    return rows


# ── results table ────────────────────────────────────────────────────────────

def print_report():
    backends = ["chroma", "graphiti"]
    data = {b: load_completed(b) for b in backends}

    def pct(rows: dict, types: list[str], key: str, derive_from: str | None = None):
        subset = [r for r in rows.values() if r["type"] in types]
        if not subset:
            return "n/a"
        if derive_from == "inverse_correct":
            val = sum(not r["correct"] for r in subset) / len(subset)
        elif key in subset[0]:
            val = sum(r[key] for r in subset) / len(subset)
        else:
            # derive staleness_caused_failure from failure_tags for old saved results
            val = sum(
                not r["correct"] and "staleness_failure" in r.get("failure_tags", [])
                for r in subset
            ) / len(subset)
        return f"{val:.0%} (n={len(subset)})"

    ALL = ["static_fact", "staleness_sensitive", "historical_belief"]
    SS  = ["staleness_sensitive"]
    HB  = ["historical_belief"]

    print("\n=== EVAL RESULTS ===\n")
    headers = ["Metric", "ChromaDB", "Graphiti"]
    rows = [
        ["Overall accuracy (30)",              pct(data["chroma"], ALL, "correct"),                      pct(data["graphiti"], ALL, "correct")],
        ["Staleness accuracy (15)",             pct(data["chroma"], SS,  "correct"),                      pct(data["graphiti"], SS,  "correct")],
        ["Staleness error rate (15)",           pct(data["chroma"], SS,  "correct", "inverse_correct"),   pct(data["graphiti"], SS,  "correct", "inverse_correct")],
        ["Staleness-caused failures (15)",      pct(data["chroma"], SS,  "staleness_caused_failure"),     pct(data["graphiti"], SS,  "staleness_caused_failure")],
        ["Critic false-approve rate (15)",      pct(data["chroma"], SS,  "critic_false_approve"),         pct(data["graphiti"], SS,  "critic_false_approve")],
        ["Historical-belief accuracy (5)",      pct(data["chroma"], HB,  "correct"),                      pct(data["graphiti"], HB,  "correct")],
    ]

    col_w = [34, 20, 20]
    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    def row_str(cells):
        return "|" + "|".join(f" {c:<{col_w[i]-2}} " for i, c in enumerate(cells)) + "|"

    print(sep)
    print(row_str(headers))
    print(sep)
    for r in rows:
        print(row_str(r))
    print(sep)
    print("\nNote: Graphiti condition requires Neo4j. Run: python -m eval.run_eval --backend graphiti")


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["chroma", "graphiti"], help="Which condition to run")
    parser.add_argument("--cap", type=int, default=None, help="Max queries to run today")
    parser.add_argument("--report", action="store_true", help="Print results table from saved runs")
    args = parser.parse_args()

    if args.report:
        print_report()
    elif args.backend:
        queries = load_queries()
        asyncio.run(run_condition(args.backend, queries, cap=args.cap))
    else:
        parser.print_help()
