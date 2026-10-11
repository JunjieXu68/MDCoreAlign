# Legacy And Advanced Controls

The normal interface exposes production `throughput`, numerical-audit `exact`,
CPU/GPU backend choice, optional ordinary RMSF and optional W5 diagnostics.

Run `mdcorealign align --help-advanced` to discover compatibility controls.
Historical `default`, `long`, `long-fast`, `alignment-only`, `fast-global`, and
`diagnostic-rich` names remain parseable so old scripts can be reproduced.
Backend auto, balanced-lite, alternate GPU solvers, representative-frame
budgets, robust-weight tuning and sparse motion-graph outputs remain advanced
or experimental. They are not separate mature product modes and are not
recommended without their original validation context.

Explicit `--mode default` retains the historical per-frame robust workflow;
omitting `--mode` now selects production throughput.
