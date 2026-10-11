"""Public parameter contracts, observed at real CPU/GPU update kernels."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest

import mdcorealign
from mdcorealign.cli import build_parser, run


@pytest.fixture
def ensemble():
    rng = np.random.default_rng(20261011)
    reference = rng.normal(size=(12, 3)) * 4.0
    coords = reference[None] + rng.normal(scale=0.15, size=(40, 12, 3))
    coords[:, 8:, 1] += 2.5 * np.sin(np.linspace(0, 2 * np.pi, 40))[:, None]
    coords += rng.normal(scale=2.0, size=(40, 1, 3))
    return coords


class KernelCalls:
    """Observe real arguments and GPU closure values; never replace a solver."""

    def __init__(self):
        self.calls = []

    def trace(self, frame, event, value):
        name = frame.f_code.co_name
        qualifying = (
            name
            in {
                "bounded_global_weight_refinement",
                "_gpu_weight_estimation",
                "estimate_representative_weights_sequential_lean",
            }
            or frame.f_code.co_qualname == "_gpu_weight_estimation.<locals>.update"
        )
        if event == "call" and qualifying:
            self.calls.append(
                {
                    "function": name,
                    **{
                        k: frame.f_locals[k]
                        for k in (
                            "delta",
                            "rule",
                            "max_iterations",
                            "sequence_continuity_strength",
                            "update_rounds",
                            "tolerance",
                            "sigma",
                            "cutoff",
                        )
                        if k in frame.f_locals
                    },
                }
            )
        return None

    def __enter__(self):
        self.previous = sys.gettrace()
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        sys.settrace(self.previous)


def require_backend(backend):
    if backend == "gpu":
        from mdcorealign.gpu_backend import gpu_available

        if not gpu_available():
            pytest.skip("Real GPU validation requires a usable CuPy/CUDA runtime")


@pytest.mark.parametrize(
    "contract,backend", [("throughput", "cpu"), ("throughput", "gpu"), ("exact", "cpu")]
)
@pytest.mark.parametrize("entrypoint", ["cli", "api"])
@pytest.mark.parametrize("delta", [None, 0.5, 2.0], ids=["default", "delta0.5", "delta2"])
def test_delta_reaches_actual_kernel(
    contract, backend, entrypoint, delta, ensemble, tmp_path, record_property
):
    require_backend(backend)
    expected = 1.0 if delta is None else delta
    with KernelCalls() as observed:
        if entrypoint == "api":
            options = {} if delta is None else {"delta": delta}
            result = mdcorealign.solve_contract(
                ensemble,
                contract=contract,
                backend=backend,
                representative_count=16,
                precision="float64",
                store_aligned=False,
                **options,
            )
            assert result.backend_used == backend
            assert np.isfinite(result.alignment.frozen_weights).all()
            assert np.isfinite(result.alignment.robust_rmsf).all()
            metadata = result.alignment.metadata
        else:
            path = tmp_path / "coords.npy"
            np.save(path, ensemble)
            out = tmp_path / "out"
            command = [
                "align",
                "--input",
                str(path),
                "--mode",
                contract,
                "--backend",
                backend,
                *(["--gpu-precision", "float64"] if backend == "gpu" else []),
                "--exact-representative-count"
                if contract == "exact"
                else "--fast-representative-count",
                "16",
                "--output-dir",
                str(out),
            ]
            if delta is not None:
                command += ["--delta", str(delta)]
            assert run(command) == 0
            metadata = json.loads((out / "diagnostics/run_metadata.json").read_text())
            with np.load(out / "transforms.npz") as transforms:
                assert np.isfinite(transforms["rotations"]).all()
    assert observed.calls
    assert all(call["delta"] == expected for call in observed.calls)
    if backend == "gpu":
        updates = [c for c in observed.calls if c["function"] == "update"]
        assert len(updates) == 4
    assert metadata["effective_parameters"]["delta"] == expected
    assert metadata["requested_parameters"]["delta"] == ("DEFAULT" if delta is None else delta)
    record_property("kernel_calls", json.dumps(observed.calls))
    record_property("effective_delta", expected)
    record_property("requested_delta", "DEFAULT" if delta is None else delta)
    record_property("backend", backend)
    record_property("entrypoint", entrypoint)


@pytest.mark.parametrize(
    "contract,backend", [("throughput", "cpu"), ("throughput", "gpu"), ("exact", "cpu")]
)
@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf"), -float("inf")])
def test_api_rejects_invalid_delta_before_any_kernel(contract, backend, bad, ensemble):
    with KernelCalls() as observed, pytest.raises(ValueError, match="finite and positive"):
        mdcorealign.solve_contract(ensemble, contract=contract, backend=backend, delta=bad)
    assert observed.calls == []


@pytest.mark.parametrize("contract", ["throughput", "exact"])
@pytest.mark.parametrize("bad", ["0", "-0.5", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_delta_before_reading_input(contract, bad, tmp_path, capsys):
    assert (
        run(
            [
                "align",
                "--input",
                str(tmp_path / "missing.npy"),
                "--mode",
                contract,
                f"--delta={bad}",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert "delta must be finite and positive" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "contract,backend", [("throughput", "cpu"), ("throughput", "gpu"), ("exact", "cpu")]
)
@pytest.mark.parametrize("entrypoint", ["cli", "api"])
def test_sequence_strength_reaches_actual_kernel(contract, backend, entrypoint, ensemble, tmp_path):
    require_backend(backend)
    with KernelCalls() as observed:
        if entrypoint == "api":
            result = mdcorealign.solve_contract(
                ensemble,
                contract=contract,
                backend=backend,
                representative_count=16,
                sequence_continuity_strength=0.3,
            )
            assert (
                result.alignment.metadata["effective_parameters"]["sequence_continuity_strength"]
                == 0.3
            )
        else:
            path = tmp_path / "coords.npy"
            np.save(path, ensemble)
            assert (
                run(
                    [
                        "align",
                        "--input",
                        str(path),
                        "--mode",
                        contract,
                        "--backend",
                        backend,
                        "--sequence-lambda",
                        "0.3",
                        "--output-dir",
                        str(tmp_path / "out"),
                    ]
                )
                == 0
            )
    assert observed.calls
    assert all(c["sequence_continuity_strength"] == 0.3 for c in observed.calls)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "options,message",
    [
        ({"rule": "gaussian"}, "only rule='huber'"),
        ({"max_iterations": 2}, "does not support max_iterations"),
        ({"warm_start": True}, "does not support warm_start"),
        ({"tolerance": 0.1}, "does not support tolerance"),
    ],
)
def test_throughput_rejects_inapplicable_api_options_before_kernel(
    backend, options, message, ensemble
):
    with KernelCalls() as observed, pytest.raises(ValueError, match=message):
        mdcorealign.solve_contract(ensemble, contract="throughput", backend=backend, **options)
    assert observed.calls == []


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--rule", "gaussian"], "only rule='huber'"),
        (["--max-iterations", "2"], "does not support max_iterations"),
        (["--warm-start"], "does not support warm_start"),
        (["--tolerance", "0.1"], "does not support tolerance"),
    ],
)
def test_throughput_rejects_inapplicable_cli_options_before_input(flags, message, tmp_path, capsys):
    assert run(["align", "--input", str(tmp_path / "missing.npy"), *flags]) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("entrypoint", ["cli", "api"])
@pytest.mark.parametrize("rule", ["huber", "gaussian", "tukey"])
def test_exact_supported_irls_options_are_forwarded(entrypoint, rule, ensemble, tmp_path):
    with KernelCalls() as observed:
        if entrypoint == "api":
            result = mdcorealign.solve_contract(
                ensemble,
                contract="exact",
                rule=rule,
                max_iterations=3,
                tolerance=1e-5,
                sequence_continuity_strength=0.2,
                delta=2.0,
            )
            assert np.isfinite(result.alignment.frozen_weights).all()
        else:
            path = tmp_path / "coords.npy"
            np.save(path, ensemble)
            assert (
                run(
                    [
                        "align",
                        "--input",
                        str(path),
                        "--mode",
                        "exact",
                        "--rule",
                        rule,
                        "--max-iterations",
                        "3",
                        "--tolerance",
                        "1e-5",
                        "--sequence-lambda",
                        "0.2",
                        "--delta",
                        "2.0",
                        "--output-dir",
                        str(tmp_path / "out"),
                    ]
                )
                == 0
            )
    assert len(observed.calls) == 1
    active_scale = {"huber": "delta", "gaussian": "sigma", "tukey": "cutoff"}[rule]
    assert observed.calls[0][active_scale] == 2.0
    assert {
        k: observed.calls[0][k]
        for k in ("rule", "max_iterations", "tolerance", "delta", "sequence_continuity_strength")
    } == {
        "rule": rule,
        "max_iterations": 3,
        "tolerance": 1e-5,
        "delta": 2.0,
        "sequence_continuity_strength": 0.2,
    }


def test_exact_rejects_disabled_chronological_warm_start(ensemble, tmp_path, capsys):
    with KernelCalls() as observed, pytest.raises(ValueError, match="chronological warm starts"):
        mdcorealign.solve_contract(ensemble, contract="exact", warm_start=False)
    assert observed.calls == []
    assert (
        run(
            [
                "align",
                "--input",
                str(tmp_path / "missing.npy"),
                "--mode",
                "exact",
                "--no-warm-start",
            ]
        )
        == 2
    )
    assert "chronological warm starts" in capsys.readouterr().err


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_throughput_accepts_explicit_independent_updates(backend, ensemble, tmp_path):
    require_backend(backend)
    result = mdcorealign.solve_contract(
        ensemble, contract="throughput", backend=backend, warm_start=False, rule="huber"
    )
    assert result.alignment.metadata["effective_parameters"]["warm_start"] is False
    path = tmp_path / "coords.npy"
    np.save(path, ensemble)
    assert (
        run(
            [
                "align",
                "--input",
                str(path),
                "--backend",
                backend,
                "--no-warm-start",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )


def test_contract_alias_overrides_only_implicit_mode(ensemble, tmp_path, capsys):
    path = tmp_path / "coords.npy"
    np.save(path, ensemble)
    for extra in [[], ["--mode", "exact"]]:
        out = tmp_path / ("alias" if not extra else "consistent")
        assert (
            run(
                [
                    "align",
                    "--input",
                    str(path),
                    "--contract",
                    "exact",
                    *extra,
                    "--output-dir",
                    str(out),
                ]
            )
            == 0
        )
        metadata = json.loads((out / "diagnostics/run_metadata.json").read_text())
        assert metadata["execution_contract"] == "exact"
    assert run(["align", "--input", str(path), "--mode", "throughput", "--contract", "exact"]) == 2
    assert "conflicting methods" in capsys.readouterr().err


def test_parser_distinguishes_defaults_from_explicit_equal_values():
    parser = build_parser()
    default = parser.parse_args(["align", "--input", "x.npy"])
    explicit = parser.parse_args(
        ["align", "--input", "x.npy", "--delta", "0.2", "--mode", "throughput"]
    )
    assert default.delta == explicit.delta == 0.2
    assert "delta" not in getattr(default, "_explicit_options", ())
    assert {"delta", "mode"} <= set(explicit._explicit_options)


@pytest.mark.parametrize("mode", ["fast-global", "diagnostic-rich"])
def test_confidence_subfits_use_requested_parameters(mode, ensemble, tmp_path):
    path = tmp_path / "coords.npy"
    np.save(path, ensemble)
    with KernelCalls() as observed:
        assert (
            run(
                [
                    "align",
                    "--input",
                    str(path),
                    "--mode",
                    mode,
                    "--fast-representative-count",
                    "16",
                    "--delta",
                    "0.5",
                    "--sequence-lambda",
                    "0.3",
                    "--output-dir",
                    str(tmp_path / "out"),
                ]
            )
            == 0
        )
    assert len(observed.calls) >= 5
    assert all(
        c["delta"] == 0.5 and c["sequence_continuity_strength"] == 0.3 for c in observed.calls
    )


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_balanced_audit_and_refinement_share_parameters(backend, ensemble):
    require_backend(backend)
    with KernelCalls() as observed:
        result = mdcorealign.solve_contract(
            ensemble,
            contract="balanced-lite",
            backend=backend,
            representative_count=16,
            audit_subset_count=8,
            audit_heldout_count=8,
            refinement_representative_count=24,
            delta=0.5,
            sequence_continuity_strength=0.3,
        )
    assert result.backend_used == backend
    assert len(observed.calls) >= 3
    assert all(
        c["delta"] == 0.5 and c["sequence_continuity_strength"] == 0.3 for c in observed.calls
    )


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_sequence_strength_is_rejected_before_kernel(backend, strength, ensemble):
    with KernelCalls() as observed, pytest.raises(ValueError, match="finite and in"):
        mdcorealign.solve_contract(
            ensemble, contract="throughput", backend=backend, sequence_continuity_strength=strength
        )
    assert observed.calls == []


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("delta", [0.5, 2.0])
def test_rigid_body_remains_rigid_with_explicit_delta(backend, delta):
    require_backend(backend)
    base = np.random.default_rng(811).normal(size=(12, 3)) * 3.0
    coords = []
    for angle in np.linspace(0.0, 1.0, 8):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        coords.append(base @ rotation.T + [angle, -2 * angle, 0.3 * angle])
    result = mdcorealign.solve_contract(
        np.asarray(coords),
        contract="throughput",
        backend=backend,
        precision="float64",
        delta=delta,
        store_aligned=True,
    )
    assert result.backend_used == backend
    np.testing.assert_allclose(result.alignment.frozen_weights, 1.0, atol=1e-12, rtol=0)
    # GPU RMSF uses E[x^2] - E[x]^2; at zero variance its roundoff is O(eps)
    # in variance, or O(sqrt(eps)) in RMSF. Bound the actual second-moment
    # cancellation from the input scale and reduction length, without tuning
    # a tolerance to the observed RMSF. Coordinates still require 1e-10 below.
    gamma = (8 * np.finfo(np.float64).eps) / (1 - 8 * np.finfo(np.float64).eps)
    variance_bound = 8 * gamma * np.max(np.sum(base * base, axis=1))
    assert np.all(result.alignment.robust_rmsf**2 <= variance_bound)
    np.testing.assert_allclose(
        result.alignment.aligned_coords, np.broadcast_to(coords[0], (8, 12, 3)), atol=1e-10, rtol=0
    )


@pytest.mark.parametrize("rule,active", [("gaussian", "sigma"), ("tukey", "cutoff")])
@pytest.mark.parametrize("delta", [None, 2.0], ids=["legacy-default", "explicit-scale"])
def test_legacy_rule_scale_preserves_default_and_honors_explicit(
    rule, active, delta, ensemble, tmp_path
):
    from mdsuperpose import weights

    kernel = getattr(weights, f"_{rule}_weights_raw").__code__
    seen = []

    def observer(frame, event, value):
        if event == "call" and frame.f_code == kernel:
            seen.append(frame.f_locals[active])
        return None

    path = tmp_path / "coords.npy"
    np.save(path, ensemble[:12])
    command = [
        "align",
        "--input",
        str(path),
        "--mode",
        "default",
        "--rule",
        rule,
        "--output-dir",
        str(tmp_path / "out"),
    ]
    if delta is not None:
        command += ["--delta", str(delta)]
    previous = sys.gettrace()
    sys.settrace(observer)
    try:
        assert run(command) == 0
    finally:
        sys.settrace(previous)
    assert seen and all(value == (1.0 if delta is None else delta) for value in seen)


def test_legacy_mode_rejects_irrelevant_update_budget(tmp_path, capsys):
    assert (
        run(
            [
                "align",
                "--input",
                str(tmp_path / "missing.npy"),
                "--mode",
                "default",
                "--fast-update-rounds",
                "8",
            ]
        )
        == 2
    )
    assert "does not support --fast-update-rounds" in capsys.readouterr().err


@pytest.mark.parametrize("delta", [0.5, 2.0])
def test_direct_public_exact_api_forwards_scale(delta, ensemble):
    with KernelCalls() as observed:
        result = mdcorealign.solve_exact_contract(ensemble, delta=delta)
    assert observed.calls and all(call["delta"] == delta for call in observed.calls)
    assert result.metadata["requested_parameters"]["delta"] == delta
    assert result.metadata["effective_parameters"]["delta"] == delta


@pytest.mark.parametrize(
    "options,match",
    [
        ({"delta": float("nan")}, "finite and positive"),
        ({"max_iterations": 0}, "at least 1"),
        ({"contact_consistency_strength": float("nan")}, "finite and in"),
    ],
)
def test_direct_public_exact_rejects_invalid_options_before_kernel(options, match, ensemble):
    with KernelCalls() as observed, pytest.raises(ValueError, match=match):
        mdcorealign.solve_exact_contract(ensemble, **options)
    assert observed.calls == []


@pytest.mark.parametrize(
    "mode,flag",
    [
        ("exact", "--fast-representative-count"),
        ("throughput", "--exact-representative-count"),
        ("balanced-lite", "--exact-representative-count"),
        ("fast-global", "--exact-representative-count"),
        ("diagnostic-rich", "--exact-representative-count"),
        ("default", "--fast-representative-count"),
        ("default", "--exact-representative-count"),
        ("long", "--fast-representative-count"),
        ("long", "--exact-representative-count"),
    ],
)
def test_wrong_method_representative_budget_rejected_before_input(mode, flag, tmp_path, capsys):
    mode_option = "--contract" if mode == "balanced-lite" else "--mode"
    with KernelCalls() as observed:
        assert (
            run(["align", "--input", str(tmp_path / "missing.npy"), mode_option, mode, flag, "16"])
            == 2
        )
    assert f"mode '{mode}' does not support {flag}" in capsys.readouterr().err
    assert observed.calls == []


def test_representative_budget_alias_and_implicit_defaults():
    from mdcorealign.cli import _prepare_parameter_contract

    parser = build_parser()
    for options in ([], ["--contract", "exact"], ["--mode", "exact"]):
        args = parser.parse_args(["align", "--input", "unused.npy", *options])
        assert args.fast_representative_count == 256
        assert args.exact_representative_count == 2000
        _prepare_parameter_contract(args)
    args = parser.parse_args(
        [
            "align",
            "--input",
            "unused.npy",
            "--contract",
            "exact",
            "--fast-representative-count",
            "256",
        ]
    )
    assert "fast_representative_count" in args._explicit_options
    with pytest.raises(ValueError, match="does not support --fast-representative-count"):
        _prepare_parameter_contract(args)


@pytest.mark.parametrize("backend", ["cpu", "auto"])
@pytest.mark.parametrize(
    "options", [{"precision": "float32"}, {"precision": "mixed"}, {"rotation_solver": "horn"}]
)
def test_cpu_resolution_rejects_incompatible_gpu_api_options_before_kernel(
    backend, options, monkeypatch, ensemble
):
    import mdcorealign.gpu_backend as gpu_backend

    monkeypatch.setattr(gpu_backend, "gpu_available", lambda: False)
    with KernelCalls() as observed, pytest.raises(ValueError, match="CPU execution supports only"):
        mdcorealign.solve_contract(ensemble, contract="throughput", backend=backend, **options)
    assert observed.calls == []


@pytest.mark.parametrize("backend", ["cpu", "auto"])
@pytest.mark.parametrize(
    "flags", [["--gpu-precision", "float64"], ["--gpu-rotation-solver", "svd"]]
)
def test_cpu_resolution_rejects_explicit_gpu_cli_controls_before_input(
    backend, flags, monkeypatch, tmp_path, capsys
):
    import mdcorealign.gpu_backend as gpu_backend

    monkeypatch.setattr(gpu_backend, "gpu_available", lambda: False)
    assert (
        run(["align", "--backend", backend, "--input", str(tmp_path / "missing.npy"), *flags]) == 2
    )
    assert "requires GPU execution; CPU uses float64 SVD" in capsys.readouterr().err


def test_auto_cpu_default_and_matching_api_policy_remain_numerically_identical(
    monkeypatch, ensemble
):
    import mdcorealign.gpu_backend as gpu_backend

    monkeypatch.setattr(gpu_backend, "gpu_available", lambda: False)
    default = mdcorealign.solve_contract(ensemble, contract="throughput", backend="auto")
    explicit = mdcorealign.solve_contract(
        ensemble, contract="throughput", backend="cpu", precision="float64", rotation_solver="svd"
    )
    assert default.backend_used == explicit.backend_used == "cpu"
    for name in ("frozen_weights", "rotations", "translations", "robust_rmsf"):
        np.testing.assert_array_equal(
            getattr(default.alignment, name), getattr(explicit.alignment, name)
        )


@pytest.mark.parametrize("backend", ["gpu", "auto"])
def test_cli_gpu_defaults_are_preserved_when_gpu_is_selected(backend, monkeypatch):
    import mdcorealign.gpu_backend as gpu_backend
    from mdcorealign.cli import _prepare_parameter_contract

    # Dispatch defaults only: real GPU arithmetic is separately traced above.
    monkeypatch.setattr(gpu_backend, "gpu_available", lambda: True)
    args = build_parser().parse_args(["align", "--input", "unused.npy", "--backend", backend])
    _prepare_parameter_contract(args)
    assert args._gpu_options == {"precision": "mixed", "rotation_solver": "horn"}
