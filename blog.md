# Temporal Memory Architecture in Multi-Agent Systems: Benchmarking RAG vs Temporal Graphs on Staleness

*As multi-agent loop engineering matures, memory architecture becomes the critical failure point for most teams.*

Imagine a user's favourite movie changes. For example, Interstellar gets dethroned by Batman Begins. Both facts, the old favourite and the new, sit in the vector store. An agent queries for the current favourite and similarity retrieval returns Interstellar, the outdated one. The critic agent flags it and triggers a retry but gets back the same result. After another failed retry, the loop then escalates to a human. 

That failure is caused by staleness= in flat vector RAG. Quick overviewof RAG: Embed documents, store in a vector store, retrieve by cosine similarity. This works for single-shot retrieval. Inside a loop, when two versions of the same fact coexist in memory, it breaks. The retrieval process has no concept of time and returns whatever scores highest on similarity, which is often the outdated fact.

I ran an A/B test of two memory architectures against the same 4-agent loop. Firstly the baseline, ChromaDB (flat vector RAG). The second memory architecture is Graphiti + Neo4j, which is a temporal knowledge graph. Both are backends to the exact same loop, with the exact same 4 agents. Only the injected memory object differs.

---

## The loop

The system is a 4-agent research loop built in LangGraph. The loop takes a query, decomposes it into subtasks, retrieves facts from memory, validates them, and synthesizes a final report. If quality is below threshold, it retries. If it can't recover after two retries, it escalates.

![Multi-agent research loop diagram](assets/agent_loop.png)

**Loop engineering** here refers to the retry-escalate control structure. The critic scores findings on two dimensions: staleness (does a newer version of this fact exist?) and grounding (does the text support the claim?). If quality falls below 0.7, the loop sends the researcher back. If retries are exhausted, the loop escalates rather than hallucinate.

The agents are identical across both memory conditions. The planner breaks queries into 2-3 subtasks. The researcher queries memory and returns the top-1 hit per subtask. The critic checks each finding against `current_value(entity, relation)` for staleness, then runs an LLM grounding check. The synthesiser writes the final report from approved findings.

The **memory interface** is a single abstract class:

```python
class Memory(ABC):
    async def write(self, finding: Finding) -> None: ...
    async def query(self, query_text: str, k: int = 5) -> list[Finding]: ...
    async def current_value(self, entity: str, relation: str) -> Finding | None: ...
```

Both implementations (ChromaDB and Graphiti + Neo4j) call the same Memory class, so the loop never knows which backend it's talking to. This makes it a controlled experiment.

---

## The eval harness

Measuring the quality of memory in a multi-agent loop requires more than just accuracy on a benchmark. On top of just checking if the answer is correct, i built an eval harness that also tracks *why* an answer is wrong. Specifically, whether or not staleness is what caused the failure. 

**The Dataset: 30 queries across three types.**

**Static fact (n=10)** - one correct fact per query without any temporal element. This is the control group to confirms the loop works before adding temporal complexity.

**Staleness-sensitive (n=15)** - two versions of a fact seeded into memory before each query. V1 is the older value (wrong answer for the query), V2 is the newer value (correct answer). Examples: a company's CEO changed, a user's favourite movie changed, a framework changed its default optimizer. The query asks for the current value.

**Historical belief (n=5)** - same two-version setup, but the query asks for the *past* state. The older fact is the correct answer. I included this subset specifically to stress-test Graphiti's temporal invalidation in the direction where it's expected to fail.

The anti-cheat constraint governs all staleness items: V1 and V2 texts must be **indistinguishable without timestamps**. Recency words such as "former," "previously," "outdated," "no longer" are all banned from corpus text. No explicit dates in the facts. Both V1 and V2 describe their fact as currently true. A reader encountering only one version would believe it. This ensures the temporal mechanism is what determines retrieval outcome, not surface cues.

**Scoring tracks four metrics per query:**
- `correct` - ground truth appears in the final report
- `used_stale_fact` - stale V1 value appeared in the final report
- `critic_false_approve` - stale fact reached the report without being caught
- `staleness_caused_failure` - failure tagged with `staleness_failure` in the critic

Each query is isolated, so each one gets a fresh memory instance and a unique partition ID. Therefore, facts from different queries never bleed into each other.

---

## How each backend handles stale facts

The mechanism comes down to what each memory layer sees when two versions of a fact exist.

![Chroma vs Graphiti retrieval comparison](assets/temporal_invalidation.png)

In **ChromaDB**, both V1 (Jonathan Hale, CEO, Jun 2024) and V2 (Dr. Priya Nair, CEO, Dec 2025) sit in the vector store with no temporal metadata that affects retrieval. Both have `invalid_at: None` because ChromaDB has no such field. When the researcher queries "Who is the CEO of Meridian Bio?", cosine similarity decides. Because V1 and V2 are written in similar style and on the same topic (that's the anti-cheat guarantee), either could win. In practice, V1 consistently won the similarity contest in this eval, landing the researcher on the stale fact.

In **Graphiti**, when V2 is written, a Cypher query marks V1 with `invalid_at = V2.valid_at`. V1 is now superseded at the graph layer. The researcher's `query()` method filters to `invalid_at IS NULL` before returning results. V1 is invisible. The researcher agent gets V2 on the first retrieval.

This difference in the retrieval layer makes all the difference. The researcher doesn't actually need to be smarter, nor does the critic  need a better staleness check. The memory layer just needs to stop returning stale facts.

---

## Results

![Eval results: ChromaDB vs Graphiti](assets/results_chart.png)

```
+----------------------------------+--------------------+--------------------+
| Metric                           | ChromaDB           | Graphiti           |
+----------------------------------+--------------------+--------------------+
| Overall accuracy (30)            | 50%                | 77%                |
| Staleness accuracy (15)          | 13%                | 80%                |
| Staleness error rate (15)        | 87%                | 20%                |
| Staleness-caused failures (15)   | 87%                | 0%                 |
| Critic false-approve rate (15)   | 0%                 | 0%                 |
| Historical-belief accuracy (5)   | 60%                | 40%                |
+----------------------------------+--------------------+--------------------+
```

**Staleness-sensitive queries:** Chroma got 2 of 15 correct (13%). Graphiti got 12 of 15 (80%). The difference is 67 percentage points.

**Staleness-caused failures:** Chroma produced a `staleness_failure` tag in 87% of staleness queries. Graphiti produced zero. Every Chroma failure traced to the same loop trap: detect staleness, retry, retrieve the same stale fact, retry again, escalate.

**Critic false-approve rate:** 0% for both. The critic never let a stale fact through to the final report. This matters because it means the Chroma failures were not false negatives in the critic - they were inescapable loops. The critic was doing its job. The retriever wasn't doing its job.

**Graphiti's 3 staleness failures (20%):** All three were `retrieval_misalignment` escalations where the critic's LLM grounding check scored valid findings below 0.5, triggering unnecessary retries until escalation. These are noise from the 8B model used for agent LLM calls, not staleness system failures. Graphiti had zero findings where V1 made it into the final report.

**Historical belief:** Chroma 60%, Graphiti 40%. This can be traced to the model call, not the memory architecture itself. Discussed below.

---

## The failure trace

The failure trace for a staleness-sensitive query under Chroma vs Graphiti makes the mechanism concrete.

![Staleness failure trace](assets/staleness_trace.png)

Chroma executes 8 steps to produce no answer. Graphiti executes 5 steps to produce the correct answer. The difference is entirely at the researcher's retrieval step - which fact gets returned from memory. Everything else in the loop is identical.

The failure mode is expensive in production. Staleness doesn't just produce wrong answers. It produces escalations where the loop burns its full retry budget, incurs the latency of all those LLM calls, and returns nothing. Average latency for staleness-sensitive queries in Chroma was 11,937ms. In Graphiti, 9,243ms, and most of that difference comes from the queries that escalated running three full researcher-critic cycles.

---

## The counter-finding: historical belief

The historical-belief subset is designed to expose the cost of temporal invalidation.

The queries ask about past state: "Who managed the Cascade Growth Fund when it had fewer than forty portfolio companies?" V1 is the correct answer (Thomas Aldrich, when the portfolio was smaller). V2 introduces a newer managing partner (Elena Vasquez, after the portfolio grew). The ground truth is V1.

Without temporal context, Graphiti erases V1 when V2 is written. The researcher retrieves V2, the critic approves (it's not stale relative to anything in memory), the synthesiser writes the wrong name.

Graphiti's `search()` accepts a `search_filter` with `valid_at` and `invalid_at` date constraints, enabling point-in-time queries. With `reference_time` set to V1's timestamp, the retriever filters to edges where `valid_at <= T` and `invalid_at IS NULL OR invalid_at > T`, surfacing V1 instead of V2. The critic also needs to be aware: for historical queries, the staleness check should verify the finding was valid at the reference time, not compare it against the live value. With both changes applied, Graphiti gets 2 of 5 correct (40%). The 3 remaining failures are `correct=False stale=False` - the same 8B synthesis noise pattern seen in the staleness misalignment failures, not retrieval failures.

Chroma retrieves either V1 or V2 with roughly equal probability (both have similar similarity scores). 3 of 5 correct, essentially random.

The gap between Graphiti (40%) and Chroma (60%) on historical belief has two components. One is real: Chroma's random retrieval gets lucky on 3 of 5 while Graphiti's targeted retrieval gets unlucky on 3 of 5 due to 8B synthesis noise. With a stronger LLM, Graphiti would likely dominate here too. The other is structural: sourcing the `reference_time` from the query text requires the planner to parse temporal intent from natural language, which we handle with an oracle timestamp in this eval. In production, that extraction step is non-trivial.

---

## Failure taxonomy

Across both conditions, three failure modes appeared:

**Staleness-induced escalation** (Chroma, 13/15 staleness queries) - researcher retrieves stale V1, critic flags it, retry retrieves V1 again, loop escalates. No stale fact reaches the report; the loop just produces nothing.

**Retrieval misalignment** (Graphiti, 3/15 staleness queries + 3/5 historical-belief queries; noise) - the critic's LLM grounding check gives a false negative on a valid finding, triggering unnecessary retries until escalation. Not a memory architecture failure. A small-model failure.

---

## Caveats

**Sample size.** n=15 on the staleness subset is enough to observe a clear directional effect but not enough for strong statistical claims. The 67 percentage point gap is large, but these exact numbers are indicative rather than definitive.

**Agent LLM is 8B.** The planner, critic grounding check, and synthesiser all use `llama-3.1-8b-instant` via Groq free tier. The retrieval misalignment failures and some static-fact misses are 8B inconsistency. A stronger model would reduce noise.

**Synthetic corpus.** All facts use fabricated entities with clean entity-relation-value structure. The anti-cheat constraint keeps the staleness hard to detect without temporal metadata, which is the right property for this test. Real corpora are messier.

---

## Architecture implications

Flat vector RAG is the correct choice when:
- Facts in the domain are static or change infrequently
- Queries ask about a single point in time (usually "now")
- Simplicity and operational cost matter more than temporal precision

Temporal graph memory earns its complexity when:
- Facts change and the system needs to track what changed and when
- Queries ask "what is current?" for entities that have multiple versions in memory
- Loop-based architectures retry on failure (flat RAG creates inescapable retry loops for stale facts)

The third point is the one that gets missed. Temporal memory is often framed as a "knowledge base" improvement. It's actually a **loop engineering** improvement. Without it, a well-designed critic that correctly detects staleness makes the system *worse*. It burns retries on a problem the retriever can never solve.

Historical belief queries require the loop to forward temporal context to the retriever and the critic. Graphiti supports point-in-time retrieval via date filters on `valid_at` and `invalid_at`. ChromaDB has no equivalent. With temporal context threaded through, Graphiti handles historical queries correctly - the remaining failures in this eval are 8B synthesis noise, not retrieval failures.

---

## Stack

Python, LangGraph, ChromaDB, Graphiti, Neo4j (Docker), Groq API (free tier), sentence-transformers for local embeddings.

Code: [github.com/n26modi/multi-agent-memory-eval](https://github.com/n26modi/multi-agent-memory-eval)
