# Agent Zamboni (`dev/zamboni`)

> **The Ice-Smoothing Methodological & Data Architect for Rinklytics (NHL Analytics)**

**Agent Zamboni** (configurable personas: **Agent Zamboni** 🚜, **Agent Gordie** 🏒, and **Agent Zambia** ⚡) is the official domain-restricted hockey analytics reference and quantitative Q&A system for Rinklytics.

## 📚 Documentation & Design Blueprints
1. **[`ZAMBONI_ARCHITECTURE_DESIGN.md`](ZAMBONI_ARCHITECTURE_DESIGN.md)**: Complete architectural design, Mermaid state/sequence/topology diagrams, single-vs-multi-agent trade-off analysis, global 1-req/sec Hugging Face rate-limiting specification, and multi-tier compute architecture (`LM Studio` local $\to$ `ZeroGPU` cloud $\to$ `zamboni-guardrails-v1`).
2. **[`KNOWLEDGE_AND_PIPELINE_SPEC.md`](KNOWLEDGE_AND_PIPELINE_SPEC.md)**: Complete calculation compendium across all 11 Rinklytics dashboard views, exact LaTeX formulas, Hugging Face dataset mappings (`jjerlee/nhl-analytics`), LangGraph state machine design, and Label Studio human-in-the-loop (HITL) curation pipeline.

## ⚡ Quick Diagnostics & CLI Usage

Always execute via the `uv` environment:

```bash
# 1. Run full diagnostic verification suite (Guardrails, 1 req/s HF Rate Limiter, RAG, Compute)
uv run python -m dev.zamboni.cli --diagnose

# 2. Ask a methodology or dataset question from the CLI
uv run python -m dev.zamboni.cli --ask "How are xG and goalie GSAx calculated using the Dual-Stage Logistic model?"

# 3. Export interaction logs for Label Studio annotation
uv run python -m dev.zamboni.cli --export-label-studio zamboni_tasks.json

# 4. Print Label Studio XML Labeling Interface
uv run python -m dev.zamboni.cli --print-label-studio-xml
```
