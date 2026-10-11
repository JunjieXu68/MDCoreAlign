# Contributing

Use a focused branch, add tests for behavioural changes, and run the canonical
test command documented in `README.md`. Do not commit large trajectories,
download caches, credentials or generated aligned-coordinate archives. Keep
CPU imports independent of CuPy and preserve explicit backend behaviour.

Scientific changes should state the output contract, units, reference choice,
atom selection and validation data. Benchmark changes must preserve frozen
evidence rather than overwrite it.

