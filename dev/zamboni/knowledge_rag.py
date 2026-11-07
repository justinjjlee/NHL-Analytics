"""
knowledge_rag.py — Hybrid Lexical + Subword TF-IDF / BM25 Retriever for Agent Zamboni.

Retrieves the most relevant Rinklytics methodology, formula, and page reference chunks
without requiring heavy GPU embedding libraries on the backend.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional

from .knowledge_base import get_canonical_knowledge_chunks
from .models import KnowledgeChunk


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric and hockey-symbol tokens."""
    if not text:
        return []
    return re.findall(r"[a-z0-9_%+-]+", text.lower())


class ZamboniKnowledgeRAG:
    """
    Lightweight, zero-dependency Hybrid BM25 + Keyword Boost Retriever over
    Rinklytics canonical knowledge chunks.
    """

    def __init__(self, chunks: Optional[List[KnowledgeChunk]] = None):
        self.chunks: List[KnowledgeChunk] = chunks or get_canonical_knowledge_chunks()
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: Counter = Counter()
        self._avg_dl: float = 1.0
        self._build_index()

    def _build_index(self) -> None:
        self._doc_tokens = []
        self._doc_freqs = Counter()
        total_len = 0

        for ch in self.chunks:
            blob = " ".join([
                ch.title * 3,
                ch.domain * 2,
                ch.page_view,
                " ".join(ch.source_files),
                " ".join(ch.hf_dataset_tables),
                ch.summary * 2,
                ch.methodology_markdown,
                " ".join(ch.keywords) * 4,
            ])
            toks = _tokenize(blob)
            self._doc_tokens.append(toks)
            total_len += len(toks)
            for ut in set(toks):
                self._doc_freqs[ut] += 1

        self._avg_dl = total_len / max(len(self.chunks), 1)

    def retrieve(
        self,
        query: str,
        target_domains: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[KnowledgeChunk]:
        """
        Scores all knowledge chunks using BM25 + exact keyword/phrase matching + domain routing boost,
        returning the top_k highest-scoring KnowledgeChunk objects.
        """
        q_low = (query or "").lower()
        q_tokens = _tokenize(q_low)
        n_docs = len(self.chunks)
        k1 = 1.5
        b = 0.75

        scored: List[KnowledgeChunk] = []
        for idx, ch in enumerate(self.chunks):
            doc_toks = self._doc_tokens[idx]
            dl = len(doc_toks)
            tf_counter = Counter(doc_toks)

            bm25_score = 0.0
            for qt in q_tokens:
                df = self._doc_freqs.get(qt, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                tf = tf_counter.get(qt, 0)
                denom = tf + k1 * (1.0 - b + b * (dl / max(self._avg_dl, 1.0)))
                bm25_score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-6))

            # Exact phrase / keyword boost
            kw_boost = 0.0
            for kw in ch.keywords:
                kw_l = kw.lower()
                if len(kw_l) <= 3:
                    if re.search(rf"\b{re.escape(kw_l)}\b", q_low):
                        kw_boost += 3.5
                elif kw_l in q_low:
                    kw_boost += 4.0

            # Domain routing boost
            domain_boost = 0.0
            if target_domains and ch.domain in target_domains:
                domain_boost = 6.0

            total_score = round(bm25_score + kw_boost + domain_boost, 3)
            copy_ch = KnowledgeChunk(
                chunk_id=ch.chunk_id,
                title=ch.title,
                domain=ch.domain,
                page_view=ch.page_view,
                source_files=list(ch.source_files),
                hf_dataset_tables=list(ch.hf_dataset_tables),
                formulas_latex=list(ch.formulas_latex),
                summary=ch.summary,
                methodology_markdown=ch.methodology_markdown,
                keywords=list(ch.keywords),
                score=total_score,
            )
            scored.append(copy_ch)

        scored.sort(key=lambda x: x.score, reverse=True)
        positives = [c for c in scored if c.score > 0]
        if not positives:
            # Default to platform overview + xG/GSAx if general hockey question
            return scored[: min(2, top_k)]
        return positives[:top_k]
