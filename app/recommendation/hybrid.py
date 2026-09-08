"""
Hybrid recommendation engine.

final_score = w_content * content_similarity
            + w_popularity * popularity_score
            + w_rating * rating_score

Design notes
------------
* Candidate generation is decoupled from ranking: we first take the top
  `candidate_pool_size` products by raw content similarity (cheap, sparse
  dot product against the whole catalog), then re-rank only that pool with
  the full hybrid score. This is the classic two-stage
  "candidate generation -> ranking" pattern, sized down for a catalog of
  tens of thousands rather than the billions it's usually used for — see
  README "Performance & Scalability" for why full pairwise similarity is
  avoided.
* All three components are normalized to [0, 1] before blending so no
  single term dominates just because of its raw scale.
* Every recommendation carries a `reason` string that names the actual
  signal that produced it (never a fabricated explanation).
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.features import text_features
from app.models.artifacts import RecommenderArtifacts


class HybridRecommender:
    def __init__(self, artifacts: RecommenderArtifacts):
        self.catalog = artifacts.catalog.reset_index(drop=True)
        self.vectorizer = artifacts.vectorizer
        self.tfidf_matrix = artifacts.tfidf_matrix
        self.popularity_score = artifacts.popularity_score.reset_index(drop=True)
        self.rating_score = (self.catalog["stars"].clip(0, 5) / 5.0).reset_index(drop=True)

        self._asin_to_row = pd.Series(
            self.catalog.index.values, index=self.catalog["asin"].values
        )

        # Documents are "title + category_<id> + bestseller" (see
        # features/text_features.build_document). That means a query with
        # ZERO real title-word overlap can still score a nonzero cosine
        # similarity purely from the category_<id>/bestseller token (and
        # any bigram that mixes a real word with one of them). If we used
        # that raw similarity to decide "does this new product have any
        # meaningful text match", we'd almost never fall back to trending
        # even for gibberish titles — the category token alone inflates it.
        # So we precompute a mask that zeroes out any vocabulary entry
        # containing "category_" or "bestseller", and use a *title-only*
        # similarity purely to decide whether to fall back. Ranking (once
        # we've decided not to fall back) still uses the full vector, since
        # "same category" IS a legitimate ranking signal — it's just not a
        # legitimate reason to claim a text match exists.
        vocab = self.vectorizer.vocabulary_
        mask = np.ones(len(vocab), dtype=np.float64)
        for token, col in vocab.items():
            if "category_" in token or "bestseller" in token:
                mask[col] = 0.0
        self._title_only_mask = mask.reshape(1, -1)
        self._title_only_matrix = self.tfidf_matrix.multiply(self._title_only_mask).tocsr()

    # ---- lookups -----------------------------------------------------
    def get_row_by_asin(self, asin: str) -> int | None:
        idx = self._asin_to_row.get(asin)
        return None if idx is None else int(idx)

    def product_exists(self, asin: str) -> bool:
        return asin in self._asin_to_row.index

    # ---- core scoring --------------------------------------------------
    def _content_scores_for_row(self, row_id: int) -> np.ndarray:
        vec = self.tfidf_matrix[row_id]
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
        return sims

    def _content_scores_for_vector(self, vec: sparse.csr_matrix) -> np.ndarray:
        sims = cosine_similarity(vec, self.tfidf_matrix).flatten()
        return sims

    def _hybrid_from_content(self, content_sim: np.ndarray, top_k: int, exclude_row: int | None):
        pool_size = min(settings.candidate_pool_size, len(content_sim))
        candidate_idx = np.argpartition(-content_sim, pool_size - 1)[:pool_size]

        pop = self.popularity_score.values[candidate_idx]
        rating = self.rating_score.values[candidate_idx]
        content = content_sim[candidate_idx]

        final = (
            settings.weight_content * content
            + settings.weight_popularity * pop
            + settings.weight_rating * rating
        )

        order = np.argsort(-final)
        ranked_candidate_idx = candidate_idx[order]
        ranked_scores = final[order]
        ranked_content = content[order]

        results = []
        for row_id, score, csim in zip(ranked_candidate_idx, ranked_scores, ranked_content):
            if exclude_row is not None and row_id == exclude_row:
                continue
            results.append((int(row_id), float(score), float(csim)))
            if len(results) >= top_k:
                break
        return results

    # ---- public API ------------------------------------------------------
    def similar_to_product(self, asin: str, top_k: int = 10):
        """Content + hybrid recommendations for an existing product (item-to-item)."""
        row_id = self.get_row_by_asin(asin)
        if row_id is None:
            return None  # caller decides cold-start fallback

        content_sim = self._content_scores_for_row(row_id)
        ranked = self._hybrid_from_content(content_sim, top_k, exclude_row=row_id)
        return self._format_results(ranked, reason_template="similar_to_product", ref_asin=asin)

    def similar_to_new_product(self, title: str, category_id: int | None, top_k: int = 10):
        """
        New-product cold start: the product itself was never in the training
        catalog (no interactions AND no fitted vector), so we vectorize its
        title on the fly with the already-fitted vectorizer and fall back to
        category + popularity if the text match is weak.
        """
        temp_df = pd.DataFrame([{
            "title": title,
            "category_id": category_id if category_id is not None else -1,
            "isBestSeller": False,
        }])
        vec = text_features.transform_new(self.vectorizer, temp_df)
        title_only_vec = vec.multiply(self._title_only_mask).tocsr()
        title_sim = cosine_similarity(title_only_vec, self._title_only_matrix).flatten()

        if title_sim.max() < 1e-6:
            # No real title-word overlap with anything in the catalog -> pure
            # popularity/category fallback rather than a ranking driven
            # entirely by the category token.
            return self.trending(top_k=top_k, category_id=category_id, reason="new_product_no_text_match")

        content_sim = self._content_scores_for_vector(vec)
        ranked = self._hybrid_from_content(content_sim, top_k, exclude_row=None)
        return self._format_results(ranked, reason_template="similar_to_new_product")

    def search(self, query: str, top_k: int = 10):
        temp_df = pd.DataFrame([{"title": query, "category_id": -1, "isBestSeller": False}])
        vec = text_features.transform_new(self.vectorizer, temp_df)
        content_sim = self._content_scores_for_vector(vec)
        ranked = self._hybrid_from_content(content_sim, top_k, exclude_row=None)
        return self._format_results(ranked, reason_template="search_match", ref_asin=query)

    def trending(self, top_k: int = 10, category_id: int | None = None, reason: str = "trending"):
        """
        Cold-start fallback for new users (no interaction history) and for
        products with no usable text signal. Pure popularity, optionally
        scoped to a category if the caller has any signal about preference
        (e.g. an onboarding question or the category of a failed lookup).
        """
        pop = self.popularity_score.copy()
        mask = pd.Series(True, index=pop.index)
        if category_id is not None:
            mask = self.catalog["category_id"] == category_id
            if mask.sum() == 0:  # unknown category -> ignore the filter
                mask = pd.Series(True, index=pop.index)

        candidates = pop[mask].sort_values(ascending=False).head(top_k)
        results = [(int(idx), float(score), None) for idx, score in candidates.items()]
        return self._format_results(results, reason_template=reason)

    def recommend_for_user(self, viewed_asins: list[str], top_k: int = 10):
        """
        New-user cold start with a *little* signal: if the user has viewed
        any products this session (but has no persisted history — this
        dataset has no user table), average their content vectors and rank
        by that; otherwise fall back to trending. This is intentionally NOT
        labeled as personalization/collaborative filtering — it is a
        session-based content heuristic and the reason string says so.
        """
        rows = [self.get_row_by_asin(a) for a in viewed_asins]
        rows = [r for r in rows if r is not None]
        if not rows:
            return self.trending(top_k=top_k, reason="new_user_no_history")

        avg_vec = self.tfidf_matrix[rows].mean(axis=0)
        avg_vec = sparse.csr_matrix(avg_vec)
        content_sim = self._content_scores_for_vector(avg_vec)
        ranked = self._hybrid_from_content(content_sim, top_k, exclude_row=None)
        ranked = [r for r in ranked if r[0] not in rows][:top_k]
        return self._format_results(ranked, reason_template="session_based_on_views")

    # ---- formatting --------------------------------------------------
    def _format_results(self, ranked, reason_template: str, ref_asin: str | None = None):
        reasons = {
            "similar_to_product": lambda: f"Similar to products you viewed ({ref_asin})",
            "similar_to_new_product": lambda: "Similar based on product title/category",
            "search_match": lambda: f"Matches your search: '{ref_asin}'",
            "trending": lambda: "Trending / popular right now",
            "new_product_no_text_match": lambda: "Popular in this category (no close text match found)",
            "new_user_no_history": lambda: "Popular right now (no browsing history yet)",
            "session_based_on_views": lambda: "Based on products you viewed this session",
        }
        reason_fn = reasons.get(reason_template, lambda: reason_template)

        out = []
        for row_id, score, content_sim in ranked:
            product = self.catalog.iloc[row_id]
            out.append({
                "asin": product["asin"],
                "title": product["title"],
                "imgUrl": product["imgUrl"],
                "price": float(product["price"]),
                "stars": float(product["stars"]),
                "category_id": int(product["category_id"]),
                "score": round(float(score), 4),
                "content_similarity": round(float(content_sim), 4) if content_sim is not None else None,
                "reason": reason_fn(),
            })
        return out
