# Multi-Agent Research Synthesis System with Temporal Memory

Full reference doc: `temporal-memory-research-system.md`. Read it at the start of a new phase.

## Stack
Python, LangGraph, FastAPI, ChromaDB, Graphiti + Neo4j, Groq API. No JavaScript. No GPU.

## Current phase
Phase 0 complete (scope locked). Starting Phase 1: Memory interface (`memory/base.py`).

## Build order
Memory seam first (Phase 1) → one backend (ChromaMemory, Phase 2a) → vertical slice end-to-end → then Graphiti, then dataset, then eval. Do not build all four agents before the seam works.

## The experimental seam (the whole point)
One `Memory` ABC. Two implementations: `ChromaMemory` (baseline) and `GraphitiMemory` (treatment). Every agent is identical across both conditions — only the injected memory object changes. That's the A/B.

## Anti-cheat checklist (gate every staleness dataset item through this)
- [ ] V1 and V2 both fluent and on-topic
- [ ] Neither contains recency words ("former", "previously", "outdated", "no longer")
- [ ] Removing the timestamp makes them genuinely indistinguishable on relevance
- [ ] A human reading only V1 would believe it (it was true once)

If any box fails, the item is too easy — rewrite it.

## Cost rules (non-negotiable)
- Agent/eval LLM calls → **Groq API** (free tier). Never route these through the Claude/Anthropic API.
- `ANTHROPIC_API_KEY` must be **unset** when using Claude Code interactively (Pro plan).
- The app loads its own `.env`; the shell stays clean.

## Key invariants
- If the final eval result is a clean one-sided Graphiti win, the staleness dataset was too easy. Return to Phase 3.
- Statistics on n≈15-20: report directional effect + bootstrap CI, not p<0.05.
- The historical-belief query subset (n≈5) is designed to *hurt* Graphiti — include it on purpose. That counter-finding is what makes the result credible.
