"""Dependency-free lexical retrieval for routing history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Sequence

from .schemas import HistoryRecord


_WORD_RE = re.compile(r"[\w]+", flags=re.UNICODE)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


@dataclass(frozen=True)
class RetrievedRecord:
    record: HistoryRecord
    similarity: float

    def to_context(self) -> dict[str, object]:
        value = self.record.to_dict()
        value["similarity"] = round(self.similarity, 6)
        return value


def tokenize(text: str) -> list[str]:
    """Tokenize words and add CJK character bigrams for useful local matching."""

    normalized = text.casefold()
    tokens = _WORD_RE.findall(normalized)
    for chunk in _CJK_RE.findall(normalized):
        if len(chunk) == 1:
            tokens.append(chunk)
        else:
            tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens


def tfidf_similarities(target: str, documents: Sequence[str]) -> list[float]:
    """Compute cosine similarities with TF-IDF fitted on this retrieval corpus."""

    if not documents:
        return []
    counters = [Counter(tokenize(text)) for text in (target, *documents)]
    document_frequency: Counter[str] = Counter()
    for counts in counters:
        document_frequency.update(counts.keys())

    count = len(counters)
    inverse_document_frequency = {
        token: math.log((1.0 + count) / (1.0 + frequency)) + 1.0
        for token, frequency in document_frequency.items()
    }

    def vector(source: Counter[str]) -> dict[str, float]:
        return {
            token: frequency * inverse_document_frequency[token]
            for token, frequency in source.items()
        }

    vectors = [vector(item) for item in counters]
    target_vector = vectors[0]
    target_norm = math.sqrt(sum(value * value for value in target_vector.values()))
    result: list[float] = []
    for candidate in vectors[1:]:
        candidate_norm = math.sqrt(sum(value * value for value in candidate.values()))
        denominator = target_norm * candidate_norm
        if denominator == 0:
            result.append(0.0)
            continue
        dot_product = sum(
            value * candidate.get(token, 0.0)
            for token, value in target_vector.items()
        )
        result.append(dot_product / denominator)
    return result


def retrieve_balanced(
    query: str,
    candidate_models: Sequence[str],
    history: Sequence[HistoryRecord],
    top_k: int,
) -> dict[str, list[RetrievedRecord]]:
    """Return independently ranked top-K records for every candidate model."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    similarities = tfidf_similarities(query, [record.query for record in history])
    grouped: dict[str, list[tuple[int, RetrievedRecord]]] = {
        model: [] for model in candidate_models
    }
    for index, (record, similarity) in enumerate(zip(history, similarities)):
        if record.model in grouped:
            grouped[record.model].append(
                (index, RetrievedRecord(record=record, similarity=similarity))
            )

    result: dict[str, list[RetrievedRecord]] = {}
    for model, records in grouped.items():
        ranked = sorted(records, key=lambda item: (-item[1].similarity, item[0]))
        result[model] = [item for _, item in ranked[:top_k]]
    return result
