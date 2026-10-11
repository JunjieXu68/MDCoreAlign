# Output Schema

`diagnostics/source_manifest.json` now uses schema version `2.2.0`. It
records source file hashes and paths, input formats, frame and atom counts,
saved fit and default export selections, coordinate-unit provenance, and
fit/frame-order fingerprints. The saved `read_configuration` also records the
actual array key, raw unit request/conversion, chain restriction and original
source frame indices. Ordered source and fit atom identities allow validated
export subsets. Source coordinate files are not copied into the result.

Coordinate export remains downstream of the authoritative compact outputs.
When all exported residues have analysis annotations, PDB/mmCIF occupancy is
`core_label`; B-factor/`B_iso_or_equiv` is `robust_RMSF` in angstrom. If a
restricted fit leaves some exported residues unanalysed, the whole export
retains source occupancy/B values and explicitly warns that it is unannotated.
mmCIF stores the complete analysed profile in
`_mdcorealign_residue_profile.*`. XTC stores coordinates only and is accompanied
by a matching topology and a small provenance JSON sidecar. The same annotation
condition applies to that topology companion.

Canonical JSON and table outputs use schema version `1.1.0`. JSON records also
contain `software_version`. This release candidate retains the established v1
RMSD/RMSF table headers and units.

## `summary.json`

Canonical throughput records exactly:

```json
{
  "method": "fixed-budget continuous-weight alignment",
  "mode": "throughput",
  "execution_contract": "throughput",
  "experimental_mode": false
}
```

`core_fraction` is the requested fraction selected by ranked continuous
stability weight. It is not a weight threshold. `effective_core_size` is the
sum of continuous weights; `effective_core_fraction` is their mean.
`fixed_update_rounds` is the configured throughput update budget. It is not a
per-frame convergence count. Exact mode instead reports measured
`average_iterations_per_frame` and `converged_fraction`.

Alignment summaries distinguish the source's default export scope from an
explicit legacy in-command coordinate export:

- `n_default_export_atoms` counts the saved default protein selection (or all
  available synthetic rows for an array).
- `n_materialized_atoms` and `materialized_selection` describe the coordinates
  actually written by `align`; they are `0` and `null` when none were written.
- `n_exportable_atoms_scope` is `default_export_selection` for analysis-only
  runs and `materialized_selection` when `align` writes coordinates.
- The compatibility field `n_exportable_atoms` follows that declared scope.
  An explicit materialized `all` selection containing four protein atoms and
  one ligand therefore reports five, while `n_default_export_atoms` remains four.

The source manifest's `n_exportable_atoms` always counts its default export
selection, independently of the optional in-command materialization. Thus the
five-atom example still permits a later normal `apply RESULT -o file.pdb` to
export the original four protein atoms. Explicit `apply` subsets are checked
against source atom identity and order; their actual counts are reported as
`n_exported_atoms` in `apply_summary.json`.

## `residue_profile.tsv`

Required v1 columns are:

```text
residue_index chain residue_id resname stable_weight mobility_score core_label
robust_RMSF mean_residual flexible_segment_id mobility_rank mobility_percentile
```

`ordinary_RMSF` is appended only with `--ordinary-rmsf`. `robust_RMSF`,
`ordinary_RMSF`, and `mean_residual` are in Angstrom. Weights, mobility and
percentiles are dimensionless.

## `frame_metrics.tsv`

`full_set_RMSD` and `weighted_RMSD` are in Angstrom and are independently
recomputed after the explicit weighted all-frame fit.

## `diagnostics/timing.json`

Canonical timing uses:

```text
weight_estimation_seconds
all_frame_fit_seconds
rmsd_seconds
rmsf_seconds
diagnostics_seconds
coordinate_materialization_seconds
end_to_end_seconds
```

Despite its name, `end_to_end_seconds` currently reports the solver total. It
does not include CLI input loading/preprocessing, report serialization or a
subsequent `apply` invocation. Unmeasured stages are not inferred from this
number. Time the complete external command separately when that is the desired
contract. `coordinate_materialization_seconds` belongs to materialization and
is not an implicit component of an analysis-only timing.

Internal v5 labels required for transition audits are isolated under
`legacy_timing` with `legacy_timing_schema="internal-v5"`; they are not
canonical fields.

## `transforms.npz`

Contains one rotation and translation per frame, frame and fit-atom counts,
fit labels, fit selection and source-unit provenance.

## Coordinate-export sidecars

`apply` writes `apply_summary.json`, `coordinate_export_provenance.json`,
`selected_frames.tsv`, `export_atom_metadata.tsv`, and `run_metadata.json`.
Coordinate materialization is separate from alignment analysis.
