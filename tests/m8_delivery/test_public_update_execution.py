"""Count real fixed-budget updates reached through the public CLI and API.

CPU updates are inline: observe their normalized-weight commit and the actual
per-round continuity function call. GPU updates are a nested function: observe
its completed calls. No solver is replaced, and metadata is not the oracle.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
import textwrap
from types import CodeType

import numpy as np
import pytest

import mdcorealign
from mdcorealign.cli import run
from mdsuperpose import window_refine


class _UpdateObserver:
    """Passive tracing of the real CPU update boundary or GPU update function."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.budgets: list[int] = []
        self.completed_rounds: list[int] = []
        self.continuity_rounds: list[int] = []
        self.weight_shapes: list[tuple[int, ...]] = []
        if backend == "cpu":
            estimator = window_refine.bounded_global_weight_refinement
            self.estimator_code = estimator.__code__
            self.update_code = window_refine.apply_sequence_continuity_regularization_batch.__code__
            source, first_line = inspect.getsourcelines(estimator)
            tree = ast.parse(textwrap.dedent("".join(source)))
            commits = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "weights"
                and isinstance(node.value, ast.Name)
                and node.value.id == "new_weights"
            ]
            assert len(commits) == 1, "Locate the actual CPU weight-update commit"
            self.commit_line = first_line + commits[0].lineno - 1
        else:
            from mdcorealign import gpu_backend

            self.estimator_code = gpu_backend._gpu_weight_estimation.__code__
            updates = [
                code
                for code in self.estimator_code.co_consts
                if isinstance(code, CodeType) and code.co_name == "update"
            ]
            assert len(updates) == 1, "Locate the actual GPU robust-update function"
            self.update_code = updates[0]
            self.commit_line = None

    def _trace(self, frame, event, value):
        if frame.f_code == self.estimator_code:
            if event == "call":
                self.budgets.append(int(frame.f_locals["update_rounds"]))
            elif self.backend == "cpu" and event == "line":
                if frame.f_lineno == self.commit_line:
                    self.completed_rounds.append(int(frame.f_locals["round_index"]))
                    weights = frame.f_locals["new_weights"]
                    assert np.isfinite(weights).all()
                    self.weight_shapes.append(weights.shape)
            return self._trace
        if frame.f_code == self.update_code and frame.f_back is not None:
            if self.backend == "cpu":
                if event == "return" and frame.f_back.f_code == self.estimator_code:
                    self.continuity_rounds.append(int(frame.f_back.f_locals["round_index"]))
            elif event == "return" and value is not None:
                # Count completed update calls; GPU loop variable `_` is reused
                # for the fit result and is deliberately not used as a counter.
                self.completed_rounds.append(len(self.completed_rounds))
                self.weight_shapes.append(value.shape)
            return self._trace
        return None

    def __enter__(self):
        self.previous_trace = sys.gettrace()
        sys.settrace(self._trace)
        return self

    def __exit__(self, *exception_info):
        sys.settrace(self.previous_trace)


@pytest.fixture
def small_ensemble() -> np.ndarray:
    """Deterministic non-rigid ensemble, with no external trajectory dependency."""
    rng = np.random.default_rng(20261011)
    reference = rng.normal(size=(12, 3)) * 4.0
    coords = reference[None, :, :] + rng.normal(scale=0.15, size=(40, 12, 3))
    coords[:, 8:, 1] += 2.5 * np.sin(np.linspace(0.0, 2.0 * np.pi, 40))[:, None]
    coords += rng.normal(scale=2.0, size=(40, 1, 3))
    return np.asarray(coords, dtype=np.float64)


@pytest.mark.parametrize("rounds", [4, 8], ids=["default-M4", "explicit-M8"])
@pytest.mark.parametrize("entrypoint", ["cli", "api"])
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_public_entrypoints_execute_requested_updates(
    backend, entrypoint, rounds, small_ensemble, tmp_path, record_property
) -> None:
    if backend == "gpu":
        from mdcorealign.gpu_backend import gpu_available

        if not gpu_available():
            pytest.skip("GPU update-count regression requires a usable CuPy/CUDA runtime")

    # Omit the parameter for M4: exercise both public defaults end to end.
    cli_budget = [] if rounds == 4 else ["--fast-update-rounds", str(rounds)]
    api_budget = {} if rounds == 4 else {"update_rounds": rounds}
    with _UpdateObserver(backend) as observed:
        if entrypoint == "cli":
            input_path = tmp_path / "ensemble.npy"
            np.save(input_path, small_ensemble, allow_pickle=False)
            output = tmp_path / "result"
            assert (
                run(
                    [
                        "align",
                        "--input",
                        str(input_path),
                        "--mode",
                        "throughput",
                        "--backend",
                        backend,
                        "--fast-representative-count",
                        "16",
                        *cli_budget,
                        *(["--gpu-precision", "float64"] if backend == "gpu" else []),
                        "--output-dir",
                        str(output),
                    ]
                )
                == 0
            )
            with np.load(output / "transforms.npz", allow_pickle=False) as transforms:
                assert np.isfinite(transforms["rotations"]).all()
                assert transforms["rotations"].shape == (40, 3, 3)
        else:
            result = mdcorealign.solve_contract(
                small_ensemble,
                contract="throughput",
                backend=backend,
                representative_count=16,
                precision="float64",
                **api_budget,
            )
            assert result.backend_used == backend
            assert np.isfinite(result.alignment.frozen_weights).all()
            assert result.alignment.frozen_weights.shape == (12,)

    record_property("backend", backend)
    record_property("entrypoint", entrypoint)
    record_property("requested_update_rounds", "DEFAULT" if rounds == 4 else str(rounds))
    record_property("expected_update_rounds", rounds)
    record_property("observed_estimator_budgets", json.dumps(observed.budgets))
    record_property("completed_update_round_indices", json.dumps(observed.completed_rounds))
    record_property("cpu_continuity_round_indices", json.dumps(observed.continuity_rounds))
    record_property("package_import_file", mdcorealign.__file__)
    assert observed.budgets == [rounds]
    assert observed.completed_rounds == list(range(rounds))
    assert observed.weight_shapes == [(12,)] * rounds
    if backend == "cpu":
        assert observed.continuity_rounds == list(range(rounds))
