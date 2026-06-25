# Multi-Agent Research Synthesis System with Temporal Memory
## Full Build Context & Instructions

> **How to use this file:** Full reference document for the project. Read top-to-bottom when starting a new phase or needing architectural depth. Day-to-day working context lives in `CLAUDE.md`.

---

## PART 0 — WHO THIS IS FOR (engineer context)

**The builder:** First-year Management Engineering student at UWaterloo (co-op), building toward an **ML Engineer career specializing in LLMs and agentic systems**, with long-term research interest. Target companies: Google, Meta, Tesla, Microsoft.

**Relevant background:**
- Strong Python (FastAPI, Pydantic, async/await). **No JavaScript/TypeScript.**
- Solid classical ML; actively building deep-learning foundations (Karpathy Zero-to-Hero done through tokenization).
- Other portfolio project: a from-scratch GRPO/RLVR reasoning model (PyTorch, Qwen2.5, GSM8K). This project is the **systems/production** counterpart to that model-depth project.
- Already researching temporal knowledge graphs (Graphiti/Zep validity-window model, recency-wins overwrite, staleness vs. correctness) for a separate startup interview — so the Graphiti angle here is intellectually live for the builder, not random.

**Tone for Claude Code:** Do not treat the builder as a beginner. Explain technically, be direct and honest, prioritize mathematical/architectural depth and *why things work*. Don't pad with generic portfolio advice. The builder cares about: building things that matter, understanding internals, getting to top companies, keeping research as a long-term option.

**What this project must achieve for the builder's narrative:** Demonstrate production AI-systems capability (agent orchestration + evaluation rigor) AND engage the current industry frontier (loop engineering, temporal-graph memory) with a genuinely novel, honest empirical finding — not a buzzword bolt-on.

---

## PART 1 — THE PROJECT IN ONE SENTENCE

> Build a verifier-gated, loop-engineered multi-agent research synthesis system, then run the **first empirical comparison of flat vector memory (ChromaDB RAG) vs. temporal knowledge-graph memory (Graphiti)** inside that system, measured specifically on **staleness-induced failure modes**.

**Why it's novel and defensible:** Multi-agent eval papers exist; temporal-KG memory exists; nobody has measured *staleness failure* as a distinct axis in a multi-agent research system via a controlled A/B where **only the memory layer changes**. It ties together three current threads — loop engineering, temporal knowledge graphs, and multi-agent failure taxonomies.

---

## PART 2 — INDUSTRY CONTEXT THAT MOTIVATES THE DESIGN

This section is background so Claude Code understands *why* the architecture is shaped this way. It does not need to be re-researched; it's settled context as of mid-2026.

### 2a. Loop engineering (the current agentic-dev frontier)
Originated from Peter Steinberger (Jun 7, 2026: "You shouldn't be prompting coding agents anymore. You should be designing loops that prompt your agents") and Boris Cherny, head of Claude Code ("I don't prompt Claude anymore. I have loops running that prompt Claude. My job is to write loops").

**A loop = four moves on repeat:** discover → plan → execute → verify → (repeat until a stop condition holds).

**Five building blocks + memory:**
1. **Triggers/automations** — a schedule/webhook fires the loop without a human pressing enter (the "heartbeat").
2. **Worktrees/isolation** — parallel agents isolated so they don't collide on shared state.
3. **Skills** — persistent project knowledge written once so agents don't re-derive intent cold.
4. **Connectors (MCP)** — let the loop *act* in real tools, not just suggest.
5. **Sub-agents (maker/checker split)** — one agent produces, a different one verifies. A model grading its own homework is too lenient.
6. **Memory/state (the spine)** — external on-disk state surviving between runs. The model forgets; the repo doesn't.

**The single most important insight:** *the verifier is the bottleneck, not the generator.* Models are cheap and strong; the generator runs nearly free. What decides whether motion produces value is the verifier — the thing that can say "no." A weak verifier doesn't fail loudly; it confidently produces garbage hundreds of times. Writing the verifier (defining "good" and "done") is the new prompt engineering and functions like a **reward function** — domain knowledge of what *correct* looks like is the moat.

**Termination logic** is mandatory: success condition (score ≥ threshold), failure cap (max N iterations, plateau/no-progress detection), escalation path (hand to human). A loop retrying the same failing action isn't learning — it's spinning.

**Known risks** (sharpen as loops improve): cost explosion, verification debt ("done" is a claim not a proof), comprehension debt, cognitive surrender.

**Scope honesty (state this in interviews):** loop engineering is currently a *coding-agent* discourse. This project applies its *principles* — verifier-gated retry, persistent state, triggers, maker/checker — to a *research-synthesis* system, a domain that hasn't gotten that treatment. That's the differentiation, not the word.

### 2b. Temporal knowledge graphs (the memory frontier)
Flat RAG (vector embeddings) retrieves on semantic similarity with **no notion of when a fact was true**. Graphiti models facts as graph edges with `valid_at`/`invalid_at` validity windows: when a newer fact about the same (entity, relation) arrives, the prior edge is marked **invalid** rather than deleted or duplicated. This is **recency-wins overwrite with no versioning** — it can answer "what's true now" but *erases* "what we believed last week." That tradeoff is itself a research-worthy question and is deliberately tested in this project (the historical-belief queries).

### 2c. Contrast point for related work — Karpathy's AutoResearch
AutoResearch (Mar 2026, viral) is a *ratchet loop*: a single coding agent edits `train.py`, runs a 5-min training job, scores against a **fixed numeric metric** (val_bpb), commits if improved, rolls back if not. It sidesteps the verifier problem by having a hard numeric ground truth. **This project's verifier is harder**: a learned critic judging unstructured research findings, where "correct" has no single numeric oracle. That contrast is the related-work framing: *unlike fixed-metric ratchet loops, multi-agent research synthesis requires a learned verifier, which introduces failure modes absent from single-metric optimization.*

---

## PART 3 — THE GUIDING PRINCIPLE (read before building anything)

The entire usefulness of this project rests on **one design decision: the staleness-injection dataset must be honest.** If superseded facts (V1) are trivially distinguishable from current facts (V2) — e.g., an explicit "OUTDATED 2019" stamp — Graphiti wins by construction and the result means nothing. The V1 fact must look **just as plausible and just as semantically relevant** as V2. The only thing separating them is *when they were true*. That is the whole point: flat vector retrieval cannot use "when," temporal-graph retrieval can.

**If the final result is a clean one-sided win for Graphiti, the dataset was too easy.** A useful finding has a *shape*: "temporal memory reduces staleness error by ~X% but costs ~Y% more latency and *fails* on historical-belief queries." The counter-finding is what makes it credible.

---

## PART 4 — ARCHITECTURE OVERVIEW

**Stack:** Python, LangGraph (stateful multi-agent graph), FastAPI (async endpoint), ChromaDB (baseline memory), Graphiti + Neo4j (treatment memory), Docker, Groq API (free-tier LLM inference). No GPU needed.

**Four agents, typed communication:**
- **Planner** — decomposes a research query into sub-tasks with a dependency graph.
- **Researcher** (multiple instances) — retrieves/analyzes via web search + the memory backend; emits structured findings with sources, confidence, and a `valid_at` timestamp.
- **Critic** — verifies findings for grounding AND staleness; requests retry if quality < threshold. This is the **verifier** — the load-bearing component.
- **Synthesiser** — aggregates approved findings into a final report with provenance.

**The experimental seam:** a single `Memory` interface with two implementations (`ChromaMemory`, `GraphitiMemory`). Every agent is byte-for-byte identical across conditions; **only the injected memory object changes.** That is what makes it a clean A/B rather than a confounded comparison.

**Loop-engineering primitives baked in:** bounded retry with plateau detection and escalate-to-human exit (termination logic); `loop_state.json` persistent state across runs; researcher/critic maker-checker split; optional cron/scheduled trigger (heartbeat).

---

## PART 5 — PHASED BUILD PLAN

### PHASE 0 — Scope lock & success criteria (½ day)
Write down "done" and "good" before coding (this is itself loop-engineering discipline).

**Done =** one FastAPI service running the 4-agent loop; two swappable memory backends behind one interface; a 50-query eval harness (15–20 staleness-sensitive) running both conditions and emitting a results table; a persistent `loop_state.json`; a short writeup with comparison table + failure taxonomy.

**Good = the findings answer:** (1) Does temporal-graph memory reduce staleness error vs flat RAG, by how much directionally? (2) Does it lower the critic's false-approve rate on stale facts? (3) What does it cost (latency, tokens, complexity)? (4) Where does recency-wins overwrite *introduce* new failures (historical-belief queries)?

---

### PHASE 1 — Memory interface abstraction (1 day)
Build the seam first so backends are truly swappable.

```python
# memory/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class Finding:
    entity: str            # "Company X"
    relation: str          # "has_CEO"
    value: str             # "Jane Doe"
    source_id: str
    confidence: float
    valid_at: float        # timestamp the source asserts this was true
    retrieved_text: str    # raw snippet, for grounding checks

class Memory(ABC):
    @abstractmethod
    def write(self, finding: Finding) -> None: ...

    @abstractmethod
    def query(self, query_text: str, k: int = 5) -> list[Finding]: ...

    @abstractmethod
    def current_value(self, entity: str, relation: str) -> Optional[Finding]:
        """Return the CURRENTLY-valid finding for (entity, relation), or None.
        ChromaMemory can only fake this; Graphiti answers it correctly.
        This asymmetry is the crux of the entire experiment."""
        ...
```

---

### PHASE 2 — The two backends (2–3 days)

**2a. ChromaMemory (baseline).** Flat vector store. `current_value` is intentionally weak — best-effort semantic match plus a *small* recency tiebreak (give the baseline this minor boost so it's not a strawman; document the choice).

```python
# memory/chroma_memory.py
import chromadb
from memory.base import Memory, Finding

class ChromaMemory(Memory):
    def __init__(self, collection_name="findings"):
        self.client = chromadb.Client()
        self.col = self.client.get_or_create_collection(collection_name)
        self._counter = 0

    def write(self, f: Finding) -> None:
        self._counter += 1
        self.col.add(
            ids=[f"f{self._counter}"],
            documents=[f.retrieved_text],
            metadatas=[{
                "entity": f.entity, "relation": f.relation, "value": f.value,
                "valid_at": f.valid_at, "source_id": f.source_id,
                "confidence": f.confidence,
            }],
        )

    def query(self, query_text: str, k: int = 5) -> list[Finding]:
        res = self.col.query(query_texts=[query_text], n_results=k)
        return [self._to_finding(m, d)
                for m, d in zip(res["metadatas"][0], res["documents"][0])]

    def current_value(self, entity, relation):
        # Flat RAG has no real notion of "current". Best-effort:
        # semantic/metadata match + naive recency tiebreak.
        # INTENTIONALLY weak — this is what we measure against.
        hits = [self._to_finding(m, d) for m, d in self._scan(entity, relation)]
        if not hits:
            return None
        return max(hits, key=lambda f: f.valid_at)
```

**2b. GraphitiMemory (treatment).** Facts as temporal edges; newer facts mark older ones `invalid_at`.

```python
# memory/graphiti_memory.py
from graphiti_core import Graphiti
from memory.base import Memory, Finding

class GraphitiMemory(Memory):
    def __init__(self, uri, user, password):
        self.g = Graphiti(uri, user, password)  # Neo4j-backed

    async def write(self, f: Finding) -> None:
        # add_episode ingests text + extracts/updates temporal edges.
        # An older (entity, relation) edge gets invalid_at set (recency-wins).
        await self.g.add_episode(
            name=f"{f.entity}:{f.relation}",
            episode_body=f.retrieved_text,
            reference_time=_to_datetime(f.valid_at),
            source_description=f.source_id,
        )

    async def query(self, query_text: str, k: int = 5) -> list[Finding]:
        results = await self.g.search(query_text, num_results=k)
        return [self._edge_to_finding(e) for e in results]

    async def current_value(self, entity, relation):
        # The payoff: filter to edges where invalid_at IS NULL (not superseded).
        edges = await self.g.get_edges(entity=entity, relation=relation)
        live = [e for e in edges if e.invalid_at is None]
        if not live:
            return None
        return self._edge_to_finding(max(live, key=lambda e: e.valid_at))
```

> **Version note:** Graphiti's exact API names shift between releases. Check the installed `graphiti-core` version's docs and adapt method names. The *shape* is what matters: ingest-with-reference-time on write, `invalid_at`-filter on read. If async/sync mismatches arise with the rest of the (sync) code, wrap appropriately or make the whole pipeline async — pick one and be consistent.

---

### PHASE 3 — The staleness dataset (THE CRITICAL PHASE) (2–3 days)
Construct a 50-query test set. **This phase decides whether the project produces a real finding.**

**Three query types:**

| Type | Count | Purpose |
|------|-------|---------|
| Static-fact | ~30 | Control. Answer doesn't change. Both backends should tie — proves Graphiti doesn't *hurt* normal queries. |
| Staleness-sensitive | ~15 | Treatment. Corpus has V1 (old, superseded) and V2 (current). Ground truth = V2. |
| Historical-belief | ~5 | Honest stress test. Asks "what was believed as of [past date]" — ground truth = V1. Graphiti's recency-wins overwrite may FAIL here. Include on purpose. |

**Constructing a staleness item honestly (example):**
```
Entity/relation: "Acme Corp" / "Series B lead investor"
V1 (valid_at 2024-03): "Acme Corp's Series B was led by Foo Ventures."
V2 (valid_at 2025-11): "Acme Corp's Series B round was re-led by Bar Capital
     after Foo Ventures exited during due diligence."
Query: "Who led Acme Corp's Series B?"
Ground truth (current): Bar Capital  (the V2 fact)

Both snippets are fluent, plausible, equally on-topic. Neither says "outdated."
The ONLY discriminator is valid_at. Flat RAG surfaces BOTH with no principled
way to prefer V2 — that's the measurable gap.
```

**Source items two ways (use both):**
1. **Real before/after pairs** — Wikipedia revision history / news (CEO changes, SOTA records broken, funding re-leads, acquired→renamed companies). Most credible.
2. **Synthetic but hand-audited** — LLM-generate plausible V1/V2 pairs, then manually verify each.

**Anti-cheat checklist — gate EVERY staleness item through this:**
- [ ] V1 and V2 both fluent and on-topic
- [ ] Neither contains recency words ("former", "previously", "outdated", "no longer")
- [ ] Removing the timestamp makes them genuinely indistinguishable on relevance
- [ ] A human reading only V1 would believe it (it was true once)

If any box fails, the item is too easy — rewrite it.

---

### PHASE 4 — The 4-agent loop with bounded retry + persistent state (2–3 days)

```python
# graph.py
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field

QUALITY_THRESHOLD = 0.7
PLATEAU_EPS = 0.03

@dataclass
class ResearchState:
    query: str
    subtasks: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    approved: list = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    last_quality: float = 0.0
    failure_tags: list = field(default_factory=list)

def route_on_quality(state: ResearchState) -> str:
    latest = score_findings(state.approved or state.findings)
    if latest >= QUALITY_THRESHOLD:
        return "approved"
    if state.retry_count >= state.max_retries:
        state.failure_tags.append("max_retries_exhausted")
        return "escalate"
    if abs(latest - state.last_quality) < PLATEAU_EPS:
        state.failure_tags.append("quality_plateau")
        return "escalate"            # stop spinning
    state.last_quality = latest
    state.retry_count += 1
    return "retry"

workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("synthesiser", synthesiser_agent)
workflow.add_node("escalate", escalate_to_human)
workflow.add_conditional_edges("critic", route_on_quality, {
    "retry": "researcher",
    "approved": "synthesiser",
    "escalate": "escalate",
})
```

**The critic's staleness check — where temporal memory pays off:**
```python
# agents/critic.py
def critic_agent(state: ResearchState, memory) -> ResearchState:
    approved, failures = [], []
    for f in state.findings:
        if not grounded(f):                          # standard grounding
            failures.append(("retrieval_misalignment", f)); continue
        current = memory.current_value(f.entity, f.relation)   # staleness check
        if current and current.valid_at > f.valid_at and current.value != f.value:
            failures.append(("staleness_failure", f)); continue
        approved.append(f)
    state.approved = approved
    state.failure_tags.extend(t for t, _ in failures)
    return state
```
Same critic code in both conditions. With `ChromaMemory`, `current_value` is weak → catches few stale facts. With `GraphitiMemory`, it catches them properly. **Change nothing in the agent — only the injected `memory` object — and measure the difference.**

**Persistent state (`loop_state.json`):**
```json
{
  "run_id": "2026-06-24T19:00Z",
  "backend": "graphiti",
  "queries": {
    "q017": {
      "type": "staleness_sensitive", "retries": 2, "final_exit": "approved",
      "failure_tags": ["staleness_failure"],
      "answer_value": "Bar Capital", "ground_truth": "Bar Capital", "correct": true
    }
  }
}
```
This turns a one-shot run into a re-runnable experiment and enables failure-drift tracking across runs.

---

### PHASE 5 — Eval harness & metrics (2 days)

```python
# eval/run_eval.py
def run_condition(backend_name, memory, queries):
    rows = []
    for q in queries:
        result = run_loop(q, memory)
        rows.append({
            "qid": q.id, "type": q.type, "backend": backend_name,
            "correct": result.answer_value == q.ground_truth,
            "used_stale_fact": result.answer_value == q.v1_value,   # key metric
            "critic_false_approve": result.bad_output_approved,
            "retries": result.retry_count, "latency_ms": result.latency_ms,
            "tokens": result.token_count, "exit": result.final_exit,
            "failure_tags": result.failure_tags,
        })
    return rows

chroma_rows   = run_condition("chroma",   ChromaMemory(),   QUERIES)
graphiti_rows = run_condition("graphiti", GraphitiMemory(...), QUERIES)
```

**Headline output table:**

| Metric | ChromaDB | Graphiti | Notes |
|---|---|---|---|
| Overall accuracy (all 50) | — | — | Should be close → proves no regression on normal queries |
| **Staleness error rate** (15–20 subset) | (expect higher) | (expect lower) | **The core finding** |
| Critic false-approve rate | — | — | Expect lower for Graphiti |
| Historical-belief accuracy (5 subset) | (expect higher) | (expect lower) | **The honest counter-finding** — recency-wins hurts here |
| Median latency / query | (lower) | (higher) | The cost |
| Median tokens / query | — | — | The cost |

**Statistics rule (bake into writeup):** at n≈15–20 on the staleness subset, report a **directional effect with effect size + bootstrap confidence interval**, NOT a p<0.05 significance claim. Overclaiming significance at this n is the #1 reviewer ding. Honest framing = stronger result.

---

### PHASE 6 — Failure taxonomy (1 day)
Existing 5 categories + the new staleness axis:
1. Planner failures — bad decomposition
2. Retrieval misalignment — wrong docs surfaced
3. Critic failures — bad approved / good rejected
4. Context/communication failures — info lost between agents
5. Latency/overflow failures
6. **Staleness failures (NEW)**:
   - 6a. *flat-RAG staleness* — backend had no temporal signal to catch it
   - 6b. *overwrite-induced staleness* — Graphiti erased a past belief a historical-belief query needed

The 6a/6b split is the intellectually load-bearing nuance: staleness isn't one problem, it's a **tradeoff between two architectures**, neither universally correct. That's what makes the work citable.

---

### PHASE 7 — Deployment + writeup (2–3 days)
- FastAPI async endpoint, Docker, structured logging. Deploy on Railway/Fly free tier.
- README as a technical design doc: architecture diagram, the memory-interface seam, the A/B design.
- arXiv-style writeup (~4–6 pages):
  - **Title:** *Temporal Knowledge-Graph Memory vs. Flat Retrieval in Multi-Agent Research Systems: An Empirical Study of Staleness Failures*
  - **Contribution sentence:** see Part 1.
  - **Related work:** position vs (a) multi-agent eval papers, (b) Graphiti/Zep temporal-KG work, (c) ratchet-loop systems with fixed numeric verifiers (Karpathy AutoResearch) — contrast in Part 2c.
  - **Limitations:** small n; synthetic staleness; recency-wins is one of several temporal models (no versioning tested).

---

## PART 6 — TIMELINE

| Phase | Days | Output |
|---|---|---|
| 0. Scope lock | 0.5 | success criteria written |
| 1. Memory interface | 1 | swappable seam |
| 2. Two backends | 2–3 | ChromaMemory + GraphitiMemory |
| 3. Staleness dataset | 2–3 | 50 queries, anti-cheat audited |
| 4. Agent loop + state | 2–3 | bounded-retry LangGraph + loop_state.json |
| 5. Eval harness | 2 | results table |
| 6. Failure taxonomy | 1 | 6-category taxonomy w/ 6a/6b split |
| 7. Deploy + writeup | 2–3 | FastAPI/Docker/Railway + arXiv draft |

**~13–17 working days.** Phases 3 and 4 are load-bearing — don't rush them.

---

## PART 7 — COMPUTE & COST SETUP

Buildable on Claude Code **Pro** ($20/mo) + Mac **M4**. No GPU anywhere. Run it as a **two-lane setup** so you never burn paid tokens on free work or vice versa.

**The M4 hosts all local infra for free:**
- **Neo4j** (backs Graphiti): JVM DB, ~2–4 GB RAM, fine on a base 16 GB M4. Install via Neo4j Desktop or `docker run neo4j`.
- **ChromaDB**: in-process, negligible.
- **Python + LangGraph + FastAPI**: trivial.
- **LLM weights**: none local — all inference via API.

**The two billing lanes (keep strictly separate):**

| Lane | For | Billing |
|---|---|---|
| **Lane 1 — Claude Code Pro ($20)** | *Writing the code* (interactive terminal coding) | Pro subscription. Grants NO API tokens; prohibits automated use. |
| **Lane 2 — Programmatic API key** | *Runtime agent LLM calls* + eval runs (thousands of calls: 4 agents × 50 queries × 2 conditions × retries) | Use **Groq free tier** → ~$0. This is your app calling an LLM, NOT Claude Code usage. |

**Critical gotcha:** if `ANTHROPIC_API_KEY` is set in your shell, **Claude Code bills pay-per-token instead of using your $20 Pro plan.** So: when *coding with Claude Code* → `ANTHROPIC_API_KEY` unset. When *running agents/eval* → keys set (Groq's, or a deliberately separate key). Keep modes separated (app loads its own `.env`; clean shell for Claude Code).

**Pro limits (mid-2026):** rolling 5-hour window (~10–45 prompts) + weekly cap; **shared across Claude Code and Claude chat**. Stay on **Sonnet** by default, use Opus deliberately. Batch coding into focused sessions. If you hit limits >~twice/week after good hygiene, that's the signal to consider Max 5x ($100) — but a paced 2–3 week project should hold on Pro.

**Cost picture:** build ≈ $20 (Pro); run ≈ $0 (Groq free tier; ~$5–20 only if agent calls are routed through Claude API instead); local DBs $0 on the M4.

**Mental model:** *Pro plan builds the thing; Groq free tier runs the thing; the M4 hosts the databases locally for free.*

---

## PART 8 — THE USEFULNESS TEST (apply before publishing)

The finding must pass all three or the dataset/design needs rework:
1. **Non-obvious** — not "fancy DB better," but "temporal memory trades staleness-robustness for latency and breaks on historical-belief queries" — a tradeoff with a shape.
2. **Actionable** — a reader learns *when* to reach for temporal-KG memory (research/monitoring agents over changing facts) and when not to (static-corpus QA, or systems needing historical belief).
3. **Honest** — includes the counter-finding (6b) and doesn't overclaim significance (Phase 5).

**If the result is a clean one-sided win, the dataset was too easy — return to Phase 3.**

---

## PART 9 — REPO STRUCTURE

```
multi-agent-memory-eval/
├── temporal-memory-research-system.md   # this file — full reference
├── CLAUDE.md                            # working session context
├── memory/
│   ├── base.py                          # Memory ABC + Finding dataclass (Phase 1)
│   ├── chroma_memory.py                 # baseline (Phase 2a)
│   └── graphiti_memory.py              # treatment (Phase 2b)
├── agents/
│   ├── planner.py
│   ├── researcher.py
│   ├── critic.py                        # the verifier — staleness check lives here
│   └── synthesiser.py
├── graph.py                             # LangGraph wiring + bounded retry (Phase 4)
├── data/
│   ├── queries.jsonl                    # 50 queries, typed (Phase 3)
│   └── corpus/                          # V1/V2 source docs for staleness items
├── eval/
│   ├── run_eval.py                      # both conditions, results table (Phase 5)
│   └── results/                         # output tables, loop_state.json
├── api/
│   └── main.py                          # FastAPI async endpoint (Phase 7)
├── Dockerfile
└── requirements.txt
```
