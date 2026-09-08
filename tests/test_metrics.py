from app.evaluation import metrics as m


def test_precision_at_k_all_relevant():
    assert m.precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_none_relevant():
    assert m.precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0


def test_precision_at_k_partial():
    assert m.precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 4) == 0.5


def test_recall_at_k_finds_all():
    assert m.recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_recall_at_k_empty_relevant_set_is_zero_not_div_by_zero():
    assert m.recall_at_k(["a", "b"], set(), 2) == 0.0


def test_hit_rate_at_k_true_when_any_hit():
    assert m.hit_rate_at_k(["a", "b", "c"], {"c"}, 3) == 1.0


def test_hit_rate_at_k_false_when_no_hit():
    assert m.hit_rate_at_k(["a", "b"], {"z"}, 2) == 0.0


def test_ndcg_perfect_ranking_is_one():
    # all top positions relevant -> NDCG should be 1.0
    assert abs(m.ndcg_at_k(["a", "b"], {"a", "b"}, 2) - 1.0) < 1e-9


def test_ndcg_penalizes_lower_rank_hits():
    high = m.ndcg_at_k(["a", "x"], {"a"}, 2)
    low = m.ndcg_at_k(["x", "a"], {"a"}, 2)
    assert high > low


def test_average_precision_rewards_early_hits():
    early = m.average_precision_at_k(["a", "x", "x"], {"a"}, 3)
    late = m.average_precision_at_k(["x", "x", "a"], {"a"}, 3)
    assert early > late


def test_reciprocal_rank_first_position():
    assert m.reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_second_position():
    assert m.reciprocal_rank(["b", "a"], {"a"}) == 0.5


def test_reciprocal_rank_no_hit_is_zero():
    assert m.reciprocal_rank(["b", "c"], {"a"}) == 0.0


def test_coverage_full_catalog_recommended():
    lists = [["a", "b"], ["c", "d"]]
    assert m.coverage(lists, catalog_size=4) == 1.0


def test_coverage_partial():
    lists = [["a"], ["a"]]
    assert m.coverage(lists, catalog_size=4) == 0.25


def test_intra_list_diversity_all_same_category_is_zero():
    assert m.intra_list_diversity([1, 1, 1]) == 0.0


def test_intra_list_diversity_all_different_is_one():
    assert m.intra_list_diversity([1, 2, 3]) == 1.0


def test_intra_list_diversity_single_item_is_zero():
    assert m.intra_list_diversity([1]) == 0.0
