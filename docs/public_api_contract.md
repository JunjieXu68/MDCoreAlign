# Stable Public API Contract

The stable Python API is the explicit `mdcorealign.__all__` list. The release
candidate intentionally does not re-export benchmark helpers, external-tool
wrappers, chronology studies, guarded research paths, internal window scanners
or private numerical helpers.

## Supported symbols

```text
ArrayCoordinateSource
ContractExecutionResult
ExactAlignmentResult
ExportAtomRecord
ExportStatistics
GPUAlignmentResult
InputReadiness
MDAnalysisCoordinateSource
SCHEMA_VERSION
__version__
build_coordinate_writer
classify_input
gpu_available
gpu_environment
parse_frame_indices
resolve_export_contract
solve_contract
solve_exact_contract
stream_apply_transforms
```

All public functions have English docstrings and complete parameter and return
annotations. Compatibility code remains importable through its defining module
but carries no stable-API guarantee.
