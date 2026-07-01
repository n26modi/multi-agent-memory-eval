# Temporal Memory Architecture in Multi-Agent Systems

Empirical A/B test comparing two memory backends in a 4-agent research loop: **ChromaDB** (flat vector RAG) vs **Graphiti + Neo4j** (temporal knowledge graph). Both backends run the same loop with identical agents. Only the injected memory object differs.

**Core finding:** temporal graph memory reduces staleness error from 87% to 20% - but introduces a 100% failure rate on historical belief queries. Full write-up in [`blog.md`](blog.md).

---

## Results

| Metric | ChromaDB | Graphiti |
|---|---|---|
| Overall accuracy (30) | 50% | 70% |
| Staleness accuracy (15) | 13% | 80% |
| Staleness error rate (15) | 87% | 20% |
| Staleness-caused failures (15) | 87% | 0% |
| Historical-belief accuracy (5) | 60% | 0% |

---

## Architecture

```
query → Planner → Researcher → Critic → Synthesiser → report
                      ↑            |
                      └── retry ───┘ (quality < 0.7, max 2 retries)
                                   └── escalate (max retries hit)
                      ↕
                   Memory
              ChromaDB | Graphiti+Neo4j
```

Four agents, one memory interface:

- **Planner** - breaks query into 2-3 subtasks
- **Researcher** - queries memory, returns top-1 hit per subtask
- **Critic** - checks findings for staleness and grounding, scores quality
- **Synthesiser** - writes final report from approved findings

The memory interface (`memory/base.py`) is a single ABC with `write`, `query`, and `current_value` methods. Two implementations: `ChromaMemory` and `GraphitiMemory`.

---

## Setup

**Requirements:** Python 3.12+, Docker (for Neo4j)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=your_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

Start Neo4j (required for Graphiti condition only):

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

---

## Running the eval

```bash
# ChromaDB condition (no Neo4j needed)
.venv/bin/python -m eval.run_eval --backend chroma

# Graphiti condition (requires Neo4j running)
.venv/bin/python -m eval.run_eval --backend graphiti

# Cap daily queries (Groq free tier)
.venv/bin/python -m eval.run_eval --backend chroma --cap 10

# Print results table from completed runs
.venv/bin/python -m eval.run_eval --report
```

Results are saved per-query to `eval/results/` and are resumable - rerunning skips completed queries.

---

## Dataset

30 queries across three types (`data/queries.jsonl`):

- **Static fact (n=10)** - one correct fact, no temporal element. Control group.
- **Staleness-sensitive (n=15)** - V1 (old value) and V2 (new value) seeded into memory. Query asks for current value. V1 and V2 are lexically indistinguishable without timestamps.
- **Historical belief (n=5)** - same two-version setup, but ground truth is V1. Designed to expose the cost of temporal invalidation.

Anti-cheat validation: `python data/anti_cheat.py`

---

## Stack

Python, LangGraph, ChromaDB, Graphiti, Neo4j, Groq API, sentence-transformers
