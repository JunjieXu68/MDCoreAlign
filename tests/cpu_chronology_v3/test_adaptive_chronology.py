"""Tests for deterministic internal adaptive-risk helpers."""

from __future__ import annotations

import numpy as np

from mdsuperpose.chronology_adaptive import AdaptiveRiskRule, adaptive_risk_features


def test_adaptive_features_are_deterministic_and_finite() -> None:
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(12, 3))
    representatives = np.stack([reference + rng.normal(scale=0.05, size=reference.shape) for _ in range(8)])
    initial = np.linspace(0.4, 1.0, 12)
    corrected = initial.copy()
    first = adaptive_risk_features(representatives, reference, initial, corrected)
    second = adaptive_risk_features(representatives, reference, initial, corrected)
    assert first == second
    assert first["core_jaccard_vs_initial"] == 1.0
    assert first["weight_spearman_vs_initial"] == 1.0
    assert all(np.isfinite(value) for value in first.values())


def test_risk_rule_is_plain_data() -> None:
    rule = AdaptiveRiskRule(0.9, 0.99, 0.0, 4.0, 0.95, 0.995, 0.0, 4.0)
    assert rule.c1_core_jaccard_min == 0.9
