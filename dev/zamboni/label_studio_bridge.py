"""
label_studio_bridge.py — Label Studio Human-in-the-Loop (HITL) Trajectory Evaluation & Curation Bridge.

Enables exporting Agent Zamboni interaction traces (`interaction_logs.parquet`) to Label Studio
annotation tasks, grading:
1. Domain Guardrail Validity
2. Tool Selection Accuracy (`pipeline_ids`)
3. Mathematical & Citation Fidelity
4. Canonical Golden Response Curation
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .dataset_store import HFZamboniStore

LABEL_STUDIO_XML_CONFIG = """<View>
  <Header value="🚜 Agent Zamboni — Trajectory Analysis, Tool Selection &amp; Golden Q&amp;A Curation"/>
  <View style="background: #f8f9fa; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
    <Text name="user_query" value="User Question: $query"/>
    <Text name="detected_intent" value="Intent: $intent | Domains: $target_domains | Pipelines: $pipeline_ids"/>
    <Text name="trajectory_meta" value="Model: $generator_model | Projected Tokens: $projected_tokens_est | Reflection Retries: $reflection_attempts | Critic Passed: $critic_passed"/>
  </View>
  <Header value="1. Hockey-Only Domain Guardrail Verification"/>
  <Choices name="guardrail_label" toName="user_query" choice="single" showInline="true">
    <Choice value="VALID_HOCKEY" alias="Valid Hockey / Rinklytics Query"/>
    <Choice value="REJECT_NON_HOCKEY" alias="Should Reject (Non-Hockey / Off-Topic)"/>
  </Choices>
  <Header value="2. Trajectory Analysis: Tool &amp; Pipeline Selection Accuracy"/>
  <Choices name="tool_selection_accuracy" toName="user_query" choice="single" showInline="true">
    <Choice value="OPTIMAL_TOOLS" alias="Optimal Pipelines Selected"/>
    <Choice value="MISSING_TOOL" alias="Missed a Relevant Skill Pipeline"/>
    <Choice value="EXTRA_TOOL" alias="Triggered Unnecessary Pipeline"/>
  </Choices>
  <Header value="3. Mathematical &amp; Citation Fidelity"/>
  <Choices name="quality_rating" toName="user_query" choice="single" showInline="true">
    <Choice value="GOLD_VERIFIED" alias="100% Accurate (Promote to Golden Cache)"/>
    <Choice value="NEEDS_EDIT" alias="Edited into Golden Response"/>
    <Choice value="HALLUCINATION" alias="Reject (Factual / Formula Drift)"/>
  </Choices>
  <Header value="4. Canonical Golden Response (Markdown + LaTeX)"/>
  <TextArea name="golden_answer" toName="user_query" editable="true" maxSubmissions="1" rows="10" transcription="true" defaultValue="$answer_markdown"/>
</View>"""


def export_logs_to_label_studio(
    output_json_path: str,
    store: HFZamboniStore | None = None,
    limit: int = 200,
) -> int:
    """Exports recent Zamboni interaction trajectories into Label Studio task JSON format."""
    store = store or HFZamboniStore()
    df = store.load_logs()
    if df.empty:
        Path(output_json_path).write_text("[]", encoding="utf-8")
        return 0

    tasks: List[Dict[str, Any]] = []
    for _, row in df.tail(limit).iterrows():
        tasks.append({
            "data": {
                "query": str(row.get("query", "")),
                "intent": str(row.get("intent", "METHODOLOGY")),
                "target_domains": str(row.get("target_domains", "[]")),
                "pipeline_ids": str(row.get("pipeline_ids", "[]")),
                "projected_tokens_est": int(row.get("projected_tokens_est", 0)),
                "reflection_attempts": int(row.get("reflection_attempts", 0)),
                "critic_passed": str(row.get("critic_passed", True)),
                "generator_model": str(row.get("generator_model", "")),
                "answer_markdown": str(row.get("answer_markdown", "")),
                "timestamp_utc": str(row.get("timestamp_utc", "")),
            }
        })

    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(tasks)


def import_annotations_from_label_studio(
    annotated_json_path: str,
    store: HFZamboniStore | None = None,
) -> int:
    """
    Parses exported Label Studio annotations JSON and promotes `GOLD_VERIFIED` or `NEEDS_EDIT`
    items into `golden_qa.parquet`.
    """
    store = store or HFZamboniStore()
    raw = json.loads(Path(annotated_json_path).read_text(encoding="utf-8"))
    golden_records: List[Dict[str, Any]] = []

    for item in raw:
        data = item.get("data", {})
        annotations = item.get("annotations", [])
        if not annotations:
            continue
        results = annotations[-1].get("result", [])
        guardrail_val = "VALID_HOCKEY"
        quality_val = None
        edited_answer = data.get("answer_markdown", "")

        for r in results:
            from_name = r.get("from_name")
            val = r.get("value", {})
            if from_name == "guardrail_label":
                choices = val.get("choices", [])
                if choices:
                    guardrail_val = choices[0]
            elif from_name == "quality_rating":
                choices = val.get("choices", [])
                if choices:
                    quality_val = choices[0]
            elif from_name == "golden_answer":
                texts = val.get("text", [])
                if texts:
                    edited_answer = texts[0]

        if guardrail_val == "VALID_HOCKEY" and quality_val in ("GOLD_VERIFIED", "NEEDS_EDIT"):
            golden_records.append({
                "query": data.get("query", ""),
                "intent": data.get("intent", "METHODOLOGY"),
                "golden_answer": edited_answer,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            })

    return store.upsert_golden_qa(golden_records)
