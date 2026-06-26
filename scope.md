# Phase 0 — Scope lock

## Done

- FastAPI service running a 4-agent loop (planner, researcher, critic, synthesiser)
- Two swappable memory backends behind one `Memory` interface (`ChromaMemory`, `GraphitiMemory`)
- 30-query eval harness running both conditions, emitting a results table
- Persistent `loop_state.json` across runs
- Blog-style article writeup with comparison table and failure taxonomy

## Good

The findings answer:

1. Does temporal-graph memory reduce staleness error vs flat RAG, and by how much directionally?
2. Does it lower the critic's false-approve rate on stale facts?
3. What does it cost (latency, tokens, complexity)?
4. Where does recency-wins overwrite introduce new failures (historical-belief queries)?

## Query set

30 total — 10 static-fact (control), 15 staleness-sensitive (treatment), 5 historical-belief (counter-finding).

## Eval exit

All 30 queries run on both conditions. Report directional effect + bootstrap CI on the staleness subset (n=15). No minimum effect size threshold.

## Writeup format

Blog/article style — architecture diagram, key finding, counter-finding (6b) given equal weight, honest caveats on small n and 8B model.
