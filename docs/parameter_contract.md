# Public Parameter Contract

All robust-loss scales (`--delta`, API `delta`) must be finite and strictly
positive, in Angstrom. Explicit invalid values fail before fitting. Omitting
delta preserves each mode's historical numerical default:

| Mode | Default delta (Angstrom) | Update policy |
| --- | --- | --- |
| throughput, fast-global, diagnostic-rich, balanced-lite | 1.0 | Fixed number of independent robust updates; default M4, explicit M8 supported |
| exact | 1.0 | Chronological sequential reference |
| legacy default, long | 0.2 | Legacy sequential policy |

Throughput supports delta and sequence continuity strength (default 0.05,
allowed range [0, 1]). It uses Huber updates. Explicit non-Huber rules,
`--max-iterations`, `--tolerance` or warm-start requests that contradict that
fixed update method are rejected. `--no-warm-start` is accepted and describes
the method's independent updates. M controls actual robust update count;
converged outputs may be equal for M4 and M8.

Exact supports its existing Huber, Tukey and Gaussian rules, delta, sequence
continuity, iteration limit, tolerance and contact controls. It is a
chronological reference, so explicit `--no-warm-start` is rejected rather than
silently ignored. `--contract exact` selects exact when no conflicting mode is
explicitly requested. An explicit contradictory mode and contract fail.

`diagnostics/run_metadata.json` records requested and effective parameter
settings. Acceptance tests trace actual CPU/GPU update calls and kernel scales;
metadata alone is not evidence of their execution. Optional GPU support is
verified only on the environments recorded in the accompanying delivery report.

Representative-frame controls are method-specific: `--fast-representative-count`
is for throughput and its fixed-budget variants; `--exact-representative-count`
is for exact. Explicitly supplying the other method's flag fails, even when its
value equals the parser default. Omitted flags preserve existing defaults.

`--gpu-precision` and `--gpu-rotation-solver` are GPU-only requests and require
actual GPU execution. Omitting them permits the usual CPU/auto fallback.
The CPU Python API implements float64/SVD; incompatible explicit precision or
rotation-solver policies fail instead of being silently discarded. This does
not add a CPU float32 or alternative rotation algorithm.
