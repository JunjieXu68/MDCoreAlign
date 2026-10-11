# Python API

MDCoreAlign exposes a deliberately small stable namespace. Research helpers and
historical compatibility modules are not re-exported from `mdcorealign`.

## Stable execution API

```python
import mdcorealign

execution = mdcorealign.solve_contract(
    coords,
    contract="throughput",
    backend="cpu",
    store_aligned=False,
)
alignment = execution.alignment
weights = alignment.frozen_weights
rotations = alignment.rotations
translations = alignment.translations
```

Use `contract="exact"` with `backend="cpu"` for the sequential reference
estimator. Use `contract="local-deformation"` for the alignment-free W5
diagnostic. An explicit GPU request requires a compatible CuPy installation and
never silently falls back to CPU.

## Stable imports

The root namespace supports the following categories:

- package and schema versions;
- `solve_contract` and `solve_exact_contract`;
- public execution result dataclasses;
- `classify_input` and its result type;
- coordinate-source, writer-selection and streaming-application entry points;
- GPU readiness inspection.

The exact symbol snapshot is maintained in `docs/public_api_contract.md` and is
checked by the release test suite.

## Coordinate application

```python
from mdcorealign import (
    ArrayCoordinateSource,
    build_coordinate_writer,
    stream_apply_transforms,
)
```

The CLI is recommended for topology-aware mmCIF, XTC and PDB output because it
also writes provenance and validates atom correspondence.

## Compatibility API

The historical `mdsuperpose` namespace remains available for research
reproducibility in this release candidate. It is not part of the stable
`mdcorealign.__all__` contract and can change in a future major or minor release.
