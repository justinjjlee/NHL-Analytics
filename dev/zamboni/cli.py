"""
cli.py — Command-Line Interface & Diagnostic Verification Tool for Agent Zamboni.

Usage:
    # Run diagnostic verification suite (guardrail, rate limiter, RAG, HF compute)
    uv run python -m dev.zamboni.cli --diagnose

    # Ask a single question via CLI
    uv run python -m dev.zamboni.cli --ask "How are xG and goalie GSAx calculated?"

    # Export interaction logs to Label Studio JSON
    uv run python -m dev.zamboni.cli --export-label-studio zamboni_tasks.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dev.zamboni.agent import ZamboniAgent
from dev.zamboni.label_studio_bridge import (
    LABEL_STUDIO_XML_CONFIG,
    export_logs_to_label_studio,
    import_annotations_from_label_studio,
)
from dev.zamboni.rate_limiter import GlobalRateLimiter


def run_diagnostics() -> None:
    print("=================================================================")
    print("🚜 AGENT ZAMBONI — DIAGNOSTIC & VERIFICATION SUITE")
    print("=================================================================")
    agent = ZamboniAgent(persona="zamboni")
    print(f"Environment Mode   : {'LOCAL (Unlimited Rate)' if agent.is_local else 'HUGGING FACE (1 req/s Global Guardrail)'}")
    print(f"LM Studio Reachable: {agent.synthesizer.lm_studio_online} (Model: {agent.synthesizer.active_lm_model})")
    print(f"Knowledge Chunks   : {len(agent.rag.chunks)} canonical domains loaded")

    # 1. Test Non-Hockey Guardrail Rejection
    print("\n[Test 1] Non-Hockey Guardrail Rejection ('How do I bake a chocolate cake?'):")
    r_bad = agent.ask("How do I bake a chocolate cake?", force_mode="algorithmic")
    print(f"  -> Allowed: {r_bad.allowed} | Error Code: {r_bad.error_code}")
    assert not r_bad.allowed and r_bad.error_code == "NON_HOCKEY_QUERY", "Non-hockey query must be rejected!"

    # 2. Test Global Rate Limiter in simulated HF mode
    print("\n[Test 2] Simulated Hugging Face Global Rate Limiter (1 req / 1.0s across all internet users):")
    sim_limiter = GlobalRateLimiter(min_interval_sec=1.0, lock_file_path="/tmp/test_zamboni_rate.lock")
    ok1, wait1 = sim_limiter.try_acquire(is_local=False)
    ok2, wait2 = sim_limiter.try_acquire(is_local=False)
    print(f"  -> Ping 1 Allowed: {ok1} (wait={wait1}s) | Immediate Ping 2 Allowed: {ok2} (wait={wait2}s)")
    assert ok1 is True and ok2 is False, "Second sub-second HF ping must be rate-limited!"

    # 3. Test Methodology + Formula Query
    print("\n[Test 3] Methodology & Formula Query ('How are xG and goalie GSAx calculated?'):")
    r_meth = agent.ask("How are xG and goalie GSAx calculated?", force_mode="algorithmic")
    print(f"  -> Allowed: {r_meth.allowed} | Intent: {r_meth.intent} | Chunks: {[c.chunk_id for c in r_meth.retrieved_chunks]} | Latency: {r_meth.latency_ms}ms")
    assert r_meth.allowed and "kb_xg_gsax_dual_stage" in [c.chunk_id for c in r_meth.retrieved_chunks]

    # 4. Test Live Hugging Face / Local Data Compute Query
    print("\n[Test 4] Live Dataset & Methodology Hybrid Query ('How is Skater PPI calculated and who are the top 5 skaters?'):")
    r_data = agent.ask("How is Skater PPI calculated and who are the top 5 skaters?", force_mode="algorithmic")
    print(f"  -> Allowed: {r_data.allowed} | Intent: {r_data.intent} | Skills: {[s.skill_name for s in r_data.computed_results]} | Latency: {r_data.latency_ms}ms")
    assert r_data.allowed and len(r_data.computed_results) >= 1

    print("\n✅ ALL AGENT ZAMBONI DIAGNOSTICS PASSED SUCCESSFULLY!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Zamboni CLI & Diagnostics")
    parser.add_argument("--diagnose", action="store_true", help="Run full verification suite")
    parser.add_argument("--ask", type=str, default=None, help="Ask Agent Zamboni a question")
    parser.add_argument("--persona", type=str, default="zamboni", choices=["zamboni", "gordie", "zambia"])
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "algorithmic"])
    parser.add_argument("--export-label-studio", type=str, default=None, help="Export logs to Label Studio JSON")
    parser.add_argument("--import-label-studio", type=str, default=None, help="Import annotated Label Studio JSON")
    parser.add_argument("--print-label-studio-xml", action="store_true", help="Print Label Studio XML config")
    args = parser.parse_args()

    if args.print_label_studio_xml:
        print(LABEL_STUDIO_XML_CONFIG)
        return

    if args.export_label_studio:
        n = export_logs_to_label_studio(args.export_label_studio)
        print(f"Exported {n} tasks to {args.export_label_studio}")
        return

    if args.import_label_studio:
        n = import_annotations_from_label_studio(args.import_label_studio)
        print(f"Imported & upserted {n} verified Golden Q&A records.")
        return

    if args.diagnose:
        run_diagnostics()
        return

    if args.ask:
        agent = ZamboniAgent(persona=args.persona)
        resp = agent.ask(args.ask, force_mode=args.mode)
        print(f"\n[{resp.generator_model} | Intent={resp.intent} | Allowed={resp.allowed} | {resp.latency_ms}ms]\n")
        print(resp.answer_markdown)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
