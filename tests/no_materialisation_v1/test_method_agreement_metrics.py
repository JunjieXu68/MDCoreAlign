"""Focused regression tests for the untimed method-agreement metrics."""

from __future__ import annotations

import numpy as np

from scripts.no_materialisation_benchmark_v1.method_agreement_metrics import (
    chainwise_jaccard,
    mask_metrics,
    segment_metrics,
    top_k_mask,
)


def test_top_k_mask_uses_exact_cardinality_and_index_tie_breaking() -> None:
    scores = np.asarray([0.8, 0.8, 0.2, 0.8, 0.1])
    mask = top_k_mask(scores, 2)
    assert mask.tolist() == [True, True, False, False, False]
    assert int(np.count_nonzero(mask)) == 2


def test_equal_cardinality_overlap_metrics_are_symmetric() -> None:
    left = np.asarray([True, True, False, False])
    right = np.asarray([True, False, True, False])
    metrics = mask_metrics(left, right)
    assert metrics["Jaccard"] == 1.0 / 3.0
    assert metrics["Dice"] == 0.5
    assert metrics["precision"] == metrics["recall"] == 0.5


def test_segment_and_chain_metrics_never_merge_across_chains() -> None:
    chains = np.asarray(["A", "A", "A", "B", "B", "B"])
    left = np.asarray([False, True, True, True, True, False])
    right = np.asarray([False, True, False, False, True, False])
    segments = segment_metrics(left, right, chains)
    chain_mean, chain_values = chainwise_jaccard(left, right, chains)
    assert segments["left_segment_count"] == 2
    assert segments["right_segment_count"] == 2
    assert np.isfinite(chain_mean)
    assert chain_values.startswith("A:") and ";B:" in chain_values
