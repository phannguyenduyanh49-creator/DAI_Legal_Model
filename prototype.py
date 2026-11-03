# -*- coding: utf-8 -*-
"""CPU-only BM25 + structured-identifier retrieval and RRF prototype."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


DEFAULT_DATASET = Path(__file__).with_name("HYBRID_RETRIEVAL_EVAL.json")
DEFAULT_OUTPUT = Path(__file__).with_name("HYBRID_RETRIEVAL_BENCHMARK.json")
IDENTIFIER_FIELDS = ("doc_number", "article", "procedure_code")


def normalize_text(text: str) -> str:
    """Normalize Vietnamese accents and punctuation for identifier matching."""
    decomposed = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def legal_tokenizer(text: str) -> list[str]:
    """Keep legal identifiers intact and also index their components."""
    decomposed = unicodedata.normalize("NFD", text.casefold().replace("đ", "d"))
    normalized = "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )
    raw_tokens = re.findall(r"[a-z0-9]+(?:[./-][a-z0-9]+)*", normalized)
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.append(token)
        if any(separator in token for separator in ("/", "-", ".")):
            tokens.extend(part for part in re.split(r"[/.-]", token) if part)
    return tokens


class BM25Engine:
    """Small, dependency-free Okapi BM25 index over document title and content."""

    def __init__(
        self,
        corpus: Sequence[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 parameters require k1 > 0 and 0 <= b <= 1")
        self.corpus = list(corpus)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [
            legal_tokenizer(f"{doc.get('title', '')} {doc.get('content', '')}")
            for doc in self.corpus
        ]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))
        self.idf = {
            term: math.log(
                1.0 + (len(self.corpus) - frequency + 0.5) / (frequency + 0.5)
            )
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not self.corpus:
            return []

        query_terms = set(legal_tokenizer(query))
        scored: list[tuple[str, float]] = []
        for doc, frequencies, doc_length in zip(
            self.corpus, self.term_frequencies, self.doc_lengths
        ):
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / self.avgdl
                )
                score += self.idf[term] * (
                    term_frequency * (self.k1 + 1) / denominator
                )
            if score > 0:
                scored.append((str(doc["doc_id"]), score))

        scored.sort(key=lambda result: (-result[1], result[0]))
        return [
            {"doc_id": doc_id, "rank": rank, "score": score}
            for rank, (doc_id, score) in enumerate(scored[:top_k], start=1)
        ]


def search_exact_metadata(
    query: str,
    corpus: Sequence[dict[str, Any]],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve records whose structured legal identifiers occur exactly."""
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if top_k == 0:
        return []

    normalized_query = f" {normalize_text(query)} "
    exact_codes = {
        normalize_text(code)
        for code in re.findall(
            r"(?<![a-z0-9])[a-z0-9]+(?:[./-][a-z0-9]+)+(?![a-z0-9])",
            query.casefold(),
        )
    }
    matches: list[tuple[int, str]] = []
    for doc in corpus:
        matched_lengths = [
            len(normalized_value)
            for field in IDENTIFIER_FIELDS
            if (value := str(doc.get(field, "")).strip())
            and (normalized_value := normalize_text(value))
            and (
                normalized_value in exact_codes
                if field == "procedure_code"
                else f" {normalized_value} " in normalized_query
            )
        ]
        if matched_lengths:
            matches.append((max(matched_lengths), str(doc["doc_id"])))

    matches.sort(key=lambda result: (-result[0], result[1]))
    return [
        {"doc_id": doc_id, "rank": rank, "score": match_length}
        for rank, (match_length, doc_id) in enumerate(matches[:top_k], start=1)
    ]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse 1-based ranked lists using classic RRF: sum(1 / (k + rank))."""
    if k <= 0:
        raise ValueError("RRF k must be positive")

    rrf_scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            doc_id = str(item["doc_id"])
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(rrf_scores.items(), key=lambda result: (-result[1], result[0]))
    return [
        {"doc_id": doc_id, "rank": rank, "rrf_score": score}
        for rank, (doc_id, score) in enumerate(ordered, start=1)
    ]


def evaluate_run(
    cases: Sequence[dict[str, Any]],
    results_by_query: dict[str, Sequence[str]],
    top_k: int,
) -> dict[str, float]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not cases:
        raise ValueError("Evaluation set must contain at least one query")

    totals = {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}
    for case in cases:
        relevant = set(case["relevant_ids"])
        if not relevant:
            raise ValueError(f"{case['query_id']} has no relevance judgments")
        ranked = list(results_by_query.get(case["query_id"], ()))[:top_k]
        hits = relevant.intersection(ranked)
        totals["recall"] += len(hits) / len(relevant)
        totals["precision"] += len(hits) / top_k

        first_relevant_rank = next(
            (rank for rank, doc_id in enumerate(ranked, start=1) if doc_id in relevant),
            None,
        )
        if first_relevant_rank is not None:
            totals["mrr"] += 1.0 / first_relevant_rank

        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(ranked, start=1)
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), top_k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        totals["ndcg"] += dcg / idcg if idcg else 0.0

    return {metric: value / len(cases) for metric, value in totals.items()}


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        dataset = json.load(file)
    corpus = dataset.get("corpus")
    cases = dataset.get("queries")
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("Dataset must contain a non-empty 'corpus' list")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Dataset must contain a non-empty 'queries' list")
    if any(not isinstance(doc, dict) or not doc.get("doc_id") for doc in corpus):
        raise ValueError("Each corpus record must be an object with a doc_id")
    if any(
        not isinstance(case, dict)
        or not case.get("query_id")
        or not case.get("query")
        or not isinstance(case.get("relevant_ids"), list)
        for case in cases
    ):
        raise ValueError("Each query must have query_id, query, and relevant_ids")
    unknown_ids = {
        doc_id
        for case in cases
        for doc_id in case["relevant_ids"]
        if doc_id not in {doc["doc_id"] for doc in corpus}
    }
    if unknown_ids:
        raise ValueError(f"Unknown relevance document IDs: {sorted(unknown_ids)}")
    return dataset


def run_benchmark(dataset: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    corpus = dataset["corpus"]
    cases = dataset["queries"]
    bm25 = BM25Engine(corpus)
    runs: dict[str, dict[str, list[str]]] = {
        "bm25": {},
        "exact_metadata": {},
        "bm25_plus_exact_rrf": {},
    }
    for case in cases:
        bm25_results = bm25.search(case["query"], top_k=top_k * 2)
        metadata_results = search_exact_metadata(
            case["query"], corpus, top_k=top_k * 2
        )
        fused_results = reciprocal_rank_fusion([bm25_results, metadata_results])
        runs["bm25"][case["query_id"]] = [item["doc_id"] for item in bm25_results]
        runs["exact_metadata"][case["query_id"]] = [
            item["doc_id"] for item in metadata_results
        ]
        runs["bm25_plus_exact_rrf"][case["query_id"]] = [
            item["doc_id"] for item in fused_results
        ]

    metrics = {
        name: evaluate_run(cases, results, top_k)
        for name, results in runs.items()
    }
    tag_metrics: dict[str, dict[str, dict[str, float]]] = {}
    for tag in sorted({tag for case in cases for tag in case.get("tags", [])}):
        tagged_cases = [case for case in cases if tag in case.get("tags", [])]
        tag_metrics[tag] = {
            name: evaluate_run(tagged_cases, results, top_k)
            for name, results in runs.items()
        }

    return {
        "classification": (
            "Synthetic closed-world prototype benchmark; not a legal-quality, "
            "production-corpus, Qdrant, or BGE-M3 measurement."
        ),
        "corpus_size": len(corpus),
        "query_count": len(cases),
        "top_k": top_k,
        "relevance": "Binary document-level judgments supplied with the sample set.",
        "metrics": metrics,
        "by_tag": tag_metrics,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    result = run_benchmark(dataset, top_k=args.top_k)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Benchmark: {result['query_count']} queries, "
        f"{result['corpus_size']} documents, K={result['top_k']}"
    )
    for name, scores in result["metrics"].items():
        print(
            f"{name:22} Recall@{args.top_k}={scores['recall']:.4f}  "
            f"Precision@{args.top_k}={scores['precision']:.4f}  "
            f"MRR={scores['mrr']:.4f}  NDCG@{args.top_k}={scores['ndcg']:.4f}"
        )
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
