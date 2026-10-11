"""Stable public Python interface for MDCoreAlign.

Research and compatibility interfaces remain available from their documented
module paths, but they are intentionally not re-exported here.
"""

from __future__ import annotations

from .contracts import (
    ContractExecutionResult,
    ExactAlignmentResult,
    solve_contract,
    solve_exact_contract,
)
from .coordinate_export import (
    ArrayCoordinateSource,
    ExportAtomRecord,
    ExportStatistics,
    MDAnalysisCoordinateSource,
    build_coordinate_writer,
    parse_frame_indices,
    resolve_export_contract,
    stream_apply_transforms,
)
from .gpu_backend import (
    GPUAlignmentResult,
    gpu_available,
    gpu_environment,
)
from .input_validation import InputReadiness, classify_input
from .output_schema import SCHEMA_VERSION

__version__ = "0.1.0rc7"

__all__ = [
    "ArrayCoordinateSource",
    "ContractExecutionResult",
    "ExactAlignmentResult",
    "ExportAtomRecord",
    "ExportStatistics",
    "GPUAlignmentResult",
    "InputReadiness",
    "MDAnalysisCoordinateSource",
    "SCHEMA_VERSION",
    "__version__",
    "build_coordinate_writer",
    "classify_input",
    "gpu_available",
    "gpu_environment",
    "parse_frame_indices",
    "resolve_export_contract",
    "solve_contract",
    "solve_exact_contract",
    "stream_apply_transforms",
]
