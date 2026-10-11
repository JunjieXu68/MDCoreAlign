"""Unambiguous stage timing records for compact CPU alignment studies."""

from __future__ import annotations

from dataclasses import dataclass
import math


STAGE_SCHEMA = {
    "end_to_end_total_seconds": (0, "", False),
    "weight_estimation_total_seconds": (1, "end_to_end_total_seconds", False),
    "candidate_initialisation_seconds": (2, "weight_estimation_total_seconds", True),
    "chronological_correction_seconds": (2, "weight_estimation_total_seconds", True),
    "blockwise_propagation_seconds": (2, "weight_estimation_total_seconds", True),
    "risk_audit_seconds": (2, "weight_estimation_total_seconds", True),
    "refinement_seconds": (2, "weight_estimation_total_seconds", True),
    "all_frame_total_seconds": (1, "end_to_end_total_seconds", False),
    "stable_rmsf_seconds": (2, "all_frame_total_seconds", True),
    "result_construction_seconds": (1, "end_to_end_total_seconds", True),
}


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Validated v3 timing values with explicit nesting semantics."""

    end_to_end_total_seconds: float
    weight_estimation_total_seconds: float
    candidate_initialisation_seconds: float = 0.0
    chronological_correction_seconds: float = 0.0
    blockwise_propagation_seconds: float = 0.0
    risk_audit_seconds: float = 0.0
    refinement_seconds: float = 0.0
    all_frame_total_seconds: float = 0.0
    stable_rmsf_seconds: float = 0.0
    result_construction_seconds: float = 0.0

    def as_dict(self) -> dict[str, float]:
        """Return values in canonical schema order."""

        return {stage: float(getattr(self, stage)) for stage in STAGE_SCHEMA}

    def validate(self, *, exact_sequential: bool = False) -> None:
        """Reject non-finite, negative, duplicated, or impossible nesting."""

        values = self.as_dict()
        for stage, value in values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{stage} must be finite and non-negative")
        tolerance = max(1e-9, self.end_to_end_total_seconds * 1e-6)
        if self.weight_estimation_total_seconds > self.end_to_end_total_seconds + tolerance:
            raise ValueError("weight-estimation parent exceeds end-to-end total")
        if self.all_frame_total_seconds > self.end_to_end_total_seconds + tolerance:
            raise ValueError("all-frame parent exceeds end-to-end total")
        weight_children = (
            self.candidate_initialisation_seconds
            + self.chronological_correction_seconds
            + self.blockwise_propagation_seconds
            + self.risk_audit_seconds
            + self.refinement_seconds
        )
        if weight_children > self.weight_estimation_total_seconds + tolerance:
            raise ValueError("additive weight children exceed their parent stage")
        if self.stable_rmsf_seconds > self.all_frame_total_seconds + tolerance:
            raise ValueError("stable RMSF child exceeds all-frame parent")
        top_level = (
            self.weight_estimation_total_seconds
            + self.all_frame_total_seconds
            + self.result_construction_seconds
        )
        if top_level > self.end_to_end_total_seconds + tolerance:
            raise ValueError("additive top-level stages exceed end-to-end total")
        if exact_sequential and self.chronological_correction_seconds != 0.0:
            raise ValueError(
                "exact sequential weight estimation must not be duplicated as a correction stage"
            )

    def rows(self, *, case_id: str, method: str, repeat: int) -> list[dict[str, object]]:
        """Render one long-format row per stage."""

        output = []
        for stage, seconds in self.as_dict().items():
            stage_level, parent_stage, is_additive = STAGE_SCHEMA[stage]
            output.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "repeat": int(repeat),
                    "stage": stage,
                    "seconds": seconds,
                    "stage_level": stage_level,
                    "parent_stage": parent_stage,
                    "is_additive": is_additive,
                }
            )
        return output


def all_frame_seconds(raw_timing: dict[str, float]) -> tuple[float, float]:
    """Map the legacy engine's local ``total`` and RMSF keys without ambiguity."""

    if "total" not in raw_timing:
        raise ValueError("all-frame engine timing lacks its local total")
    total = float(raw_timing["total"])
    rmsf = float(raw_timing.get("RMSF", 0.0))
    if total < 0.0 or rmsf < 0.0 or rmsf > total + max(1e-9, total * 1e-6):
        raise ValueError("invalid all-frame timing nesting")
    return total, rmsf
