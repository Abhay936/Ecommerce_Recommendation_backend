"""
Ranking metrics for recommendation evaluation.

Each function takes:
  recommended: list of item ids, in ranked order (best first)
  relevant: set of item ids considered "relevant" for this query
  k: cutoff

These are standard information-retrieval definitions; see the docstring
of `evaluate.py` for what "relevant" means in this project (there is no
real user feedback in the dataset, so relevance is a documented proxy).
"""
import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def average_precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """AP@K — used to compute MAP@K by averaging across queries."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = 0
    score = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    top_k = recommended[:k]
    dcg = sum(
        (1.0 if item in relevant else 0.0) / np.log2(i + 1)
        for i, item in enumerate(top_k, start=1)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(recommended: list, relevant: set) -> float:
    for i, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def coverage(all_recommended_lists: list[list], catalog_size: int) -> float:
    """Fraction of the catalog that appears at least once across all recommendation lists."""
    recommended_union = set()
    for lst in all_recommended_lists:
        recommended_union.update(lst)
    return len(recommended_union) / catalog_size if catalog_size else 0.0


def intra_list_diversity(recommended_categories: list) -> float:
    """
    1 - (fraction of pairs in the list sharing the same category).
    Higher = more diverse. Uses category_id as a coarse proxy for
    "topically different" since we have no richer taxonomy or embeddings
    to compute pairwise dissimilarity from.
    """
    n = len(recommended_categories)
    if n < 2:
        return 0.0
    same = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if recommended_categories[i] == recommended_categories[j]:
                same += 1
    return 1 - (same / total) if total else 0.0
