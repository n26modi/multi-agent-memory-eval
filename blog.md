# Does temporal graph memory fix staleness in multi-agent research loops?

I built a small empirical test to find out. The short answer: yes, dramatically - but it breaks something else in the process.

---

## The problem

Multi-agent research systems retrieve facts from memory, reason over them, and synthesize a final answer. The retrieval step is almost always a flat vector search: embed the query, find the closest chunks, return the top-k.

The problem is that flat vector stores have no concept of time. If your memory contains two facts about the same entity - one from six months ago and one from last week - the retriever picks whichever one scores higher on cosine similarity. That could easily be the older one, especially if both are written in similar language. The agent never knows it retrieved something stale.

This is a known limitation and usually handled through metadata filtering, recency reranking, or just hoping the LLM figures it out. None of these are great. So I wanted to test a structural alternative: what if the memory layer itself tracked time and invalidated old facts when new ones arrived?

---

## What I built

A 4-agent research loop in LangGraph with a pluggable memory backend:

- **Planner** - breaks the user query into 2-3 focused subtasks
- **Researcher** - queries memory for each subtask, returns the top hit
- **Critic** - checks each finding for staleness and grounding, scores quality
- **Synthesiser** - writes a final report from the approved findings

The loop retries up to 2 times if the critic's quality score is below 0.7, then escalates if it still can't get approved findings.

The key design choice is a single `Memory` ABC with two implementations that are otherwise identical from the agents' perspective:

```python
class Memory(ABC):
    async def write(self, finding: Finding) -> None: ...
    async def query(self, query_text: str, k: int = 5) -> list[Finding]: ...
    async def current_value(self, entity: str, relation: str) -> Finding | None: ...
```

- **ChromaDB** (baseline) - flat in-memory vector store, semantic similarity search, no temporal filtering
- **Graphiti + Neo4j** (treatment) - temporal knowledge graph, edges carry `valid_at` and `invalid_at` timestamps, invalidates old facts when new ones arrive

Every agent is identical across both conditions. Only the injected memory object changes. That's the A/B.

---

## The dataset

30 queries across three types:

**Static fact (n=10)** - one correct fact, no updates, no temporal element. Control group to confirm the loop works at all.

**Staleness-sensitive (n=15)** - two versions of a fact exist in memory: V1 (older, wrong answer for the query) and V2 (newer, correct answer). Examples: a company that changed its CEO, a fund that changed its lead investor, a framework that shipped a new default. The eval seeds both into memory and asks which is current.

**Historical belief (n=5)** - same two-version setup, but the query asks for the *past* state. The older fact is the correct answer. This is designed to hurt Graphiti. If temporal invalidation is too aggressive, it erases exactly the fact you needed.

The anti-cheat constraint across all staleness items: V1 and V2 texts must be indistinguishable without timestamps. No words like "former," "previously," "outdated," or explicit dates in the corpus text. Both texts describe the fact as currently true. A human reading only one version would believe it. This ensures the test is actually hard and that the temporal mechanism - not surface lexical cues - is what determines the outcome.

---

## Results

```
+----------------------------------+--------------------+--------------------+
| Metric                           | ChromaDB           | Graphiti           |
+----------------------------------+--------------------+--------------------+
| Overall accuracy (30)            | 50%                | 70%                |
| Staleness accuracy (15)          | 13%                | 80%                |
| Staleness error rate (15)        | 87%                | 20%                |
| Staleness-caused failures (15)   | 87%                | 0%                 |
| Critic false-approve rate (15)   | 0%                 | 0%                 |
| Historical-belief accuracy (5)   | 60%                | 0%                 |
+----------------------------------+--------------------+--------------------+
```

---

## What's actually happening in each condition

### Chroma: correct detection, no escape

Chroma's failure mode is a specific pattern that repeated in 13 of 15 staleness queries.

The critic is doing its job. It calls `current_value(entity, relation)` after the researcher returns a finding, compares timestamps, and correctly flags V1 as stale when V2 exists with a newer `valid_at`. Zero critic false-approvals - it never let a stale fact through.

The problem is what happens next. On retry, the researcher queries the same memory with the same or a rephrased subtask. Chroma has no temporal filtering. V1 and V2 are both in the vector store, both semantically similar to the query (that's the anti-cheat guarantee). The top-1 hit is V1 again. The critic flags it again. After two retries, the loop escalates and produces no answer.

The staleness is detectable but inescapable. The researcher can't get to V2 because the retriever doesn't know V2 is preferred.

### Graphiti: temporal filtering breaks the loop

When V2 is written into Graphiti after V1, the write method runs a Cypher query to set `invalid_at = V2.valid_at` on any live V1 edges for that entity and relation. V1 is now marked as superseded in the graph.

The researcher's `query()` fetches results from vector search and filters to `invalid_at IS NULL` before returning. V1 is excluded. V2 is the only thing the researcher sees. On first retrieval it gets the right fact, the critic approves, the synthesiser writes the report. 12 of 15 staleness queries resolved on the first pass.

The 3 Graphiti staleness failures were all `retrieval_misalignment` escalations - the critic's grounding LLM check (a separate call to score whether the finding text supports itself) gave false negatives and the loop escalated. These are LLM noise from the small 8B model used for agent calls, not staleness system failures. Graphiti had zero facts where V1 made it into the final report.

### Historical belief: Graphiti's blind spot

For historical-belief queries the correct answer is V1 - the older fact. Graphiti erased it. The researcher gets V2 (the newer, incorrect answer for this query type), the critic approves it (it's not stale relative to anything), and the synthesiser writes a wrong report. 0 out of 5.

Chroma got 3 of 5. Because it has no temporal preference, it sometimes retrieved V1 by chance.

This is the counter-finding that makes the result credible. Graphiti's temporal invalidation is a feature when you want the most current fact and a bug when you want the historical one. A system that knew the query intent could route to the right behavior. A system that always overwrites doesn't have that option.

---

## Honest caveats

**Small n.** 15 staleness queries is enough to see a directional effect but not enough to make strong statistical claims. Bootstrap CI on Graphiti's 80% staleness accuracy gives a wide interval. The effect size is large (67 percentage point difference) so the direction is clear, but the exact numbers should be treated as indicative.

**Manual invalidation vs native Graphiti.** Graphiti's `add_episode` method uses an LLM internally to extract entities and resolve temporal conflicts. That LLM call requires ~18K tokens per episode, which exceeds the token-per-minute limit on every Groq free-tier model. I replaced `add_episode` with direct node and edge construction via `add_nodes_and_edges_bulk`, managing invalidation manually with a Cypher query. The temporal semantics are preserved but Graphiti's LLM-driven entity resolution (which can merge references to the same entity across different phrasings) is not used. The eval's entity and relation strings are exact-match across V1 and V2, so this doesn't affect results here, but it's worth knowing in a messier real-world scenario.

**Agent LLM is 8B.** The planner, critic grounding check, and synthesiser all use `llama-3.1-8b-instant` via Groq. The 3 Graphiti retrieval_misalignment failures and some of the static-fact misses are almost certainly 8B inconsistency, not system design issues. A stronger model would reduce noise.

**Synthetic corpus.** All facts are fabricated entities (Meridian Bio, Cascade Growth Fund, etc.) with clean entity-relation-value structure. Real retrieval corpora are messier. The anti-cheat constraint keeps V1 and V2 hard to distinguish, which is the right property for testing temporal recall, but the overall setup is cleaner than production.

---

## What this suggests

If your agent system operates over a domain where facts change over time - personnel, prices, software versions, funding rounds, regulatory status - flat vector retrieval has a structural weakness that prompting alone doesn't fix. The critic can detect staleness but the retriever can't escape it.

Temporal graph memory addresses this at the retrieval layer rather than the reasoning layer. The agent doesn't need to be smarter about staleness; the memory just stops returning stale facts.

The tradeoff is historical queries. If your use case ever needs to answer "what was true before the update," temporal invalidation makes that harder, not easier. You'd need either a separate retrieval path that can query across time windows, or a way to signal query intent before hitting memory.

Neither backend is universally better. The right choice depends on what kinds of questions your system needs to answer.

---

## Code

Everything is on GitHub: [multi-agent-memory-eval](https://github.com/n26modi/multi-agent-memory-eval)

Stack: Python, LangGraph, ChromaDB, Graphiti, Neo4j (Docker), Groq API, sentence-transformers.
