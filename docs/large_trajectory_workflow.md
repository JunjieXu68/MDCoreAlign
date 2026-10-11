# Large-Trajectory Workflow

1. Unwrap/nojump and center the trajectory upstream with a topology-aware MD tool.
2. Run `mdcorealign audit-preprocessing` and inspect warnings.
3. Start with CA fitting and the first-frame reference.
4. Run throughput CPU or GPU without coordinate materialization.
5. Inspect `diagnostics.json`, weights, RMSF and flexible segments.
6. For critical systems, compare a selected run against `--mode exact`.
7. Run `mdcorealign apply` only when aligned coordinates are needed.

```bash
mdcorealign align --topology top.tpr --trajectory traj.xtc \
  --mode throughput --backend gpu --output-dir analysis
mdcorealign apply --topology top.tpr --trajectory traj.xtc \
  --transforms analysis/transforms.npz --apply-selection protein \
  --output-dir materialized
```

Analysis, raw-source loading and output materialization are different timing
contracts and should be reported separately. MDCoreAlign sanity checks cannot
independently prove that PBC/nojump preprocessing was correct.

Throughput approximates stable-weight estimation only. Once weights are
frozen, every frame receives an explicit weighted rigid fit. An explicit GPU
request fails with actionable guidance if CUDA/CuPy is unavailable; it does not
silently fall back to CPU. `--backend auto` remains experimental.

The analysis loader currently materializes selected trajectory coordinates.
`--chunk-size` controls numerical fit/export batching; it does not make the
whole workflow a bounded-memory streaming reader. Plan memory for the selected
frame and atom array. Solver `end_to_end_seconds` excludes input reading,
result serialization and the separate `apply` command. No new complete-command
timer or fully streaming analysis guarantee is introduced in this candidate.
