# Modes And Quality Contracts

| Workflow | Purpose | Status | GPU |
| --- | --- | --- | --- |
| `align` / `--mode throughput` | Production fixed-budget stable-weight estimation plus explicit all-frame fitting | Public production default | CPU or GPU |
| `align --mode exact` | Chronological float64 numerical reference/audit | Public audit path | CPU |
| `local-deformation` | Alignment-free W5 local internal-deformation diagnostic | Optional diagnostic | CPU or GPU |
| `apply` | Bounded-memory coordinate materialization from saved transforms | Public utility | CPU |

Canonical throughput metadata uses `method="fixed-budget continuous-weight
alignment"`, `mode="throughput"`, `execution_contract="throughput"`,
`experimental_mode=false`, and
`convergence_interpretation="not_applicable_fixed_budget"`. A fixed update
budget is neither a convergence success nor a convergence failure. Exact
reports its measured iteration count and convergence fraction.

Throughput approximates continuous stability-weight estimation only. After weights are
frozen, every frame receives an explicit weighted Kabsch fit. Exact is a
numerical reference, not biological ground truth. W5 answers a complementary
local-geometry question and is not RMSD.

Backend auto, balanced-lite, sparse motion graphs, and historical mode names
remain experimental or compatibility-only and are excluded from first-screen
help.
