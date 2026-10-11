# Changelog

## 0.1.0rc6 - Interface and distribution candidate

- Forwarded explicit robust-loss scales through CPU/GPU public entry points,
  validated finite positive values, and made explicit option handling mode-aware.
- Preserved M4 defaults and verified actual M8 update execution.
- Repaired chain filtering, input replay units and array keys, format overrides,
  source identity checks for export subsets, and synthetic UNK atom metadata.
- Unified protein selection for supported PDB and trajectory inputs.
- Retained the direct/sdist wheel runtime contract and included the two compact
  XTC example fixtures in the source distribution.
- Separated portable software regression from fixture-dependent archive QA
  without disabling archive assertions.
- This is a release candidate. Consult the accompanying RELEASE_READINESS.md
  for the actual commit, tested environments and publication status.

## 0.1.0rc5 - M8 delivery candidate

- Preserved the current CPU/GPU support for eight throughput update rounds;
  the default remains four rounds and the numerical implementation is unchanged.
- Updated the tested-rounds CLI notice and added public CLI/API regression
  checks that count actual updates for the default and explicit M8 settings.
- Applied the same research-module exclusion to direct wheel and sdist-derived
  wheel builds, with a regression check on actual build output.
- Assigned a new artifact identity. Historical VT02.2 / 0.1.0rc4 packages are
  retained and do not contain the M8 support in this candidate.
- Known issue: custom `--delta` / API delta is not reliably applied by throughput
  entry points. This candidate does not fix delta and is not a stable release.

## 0.1.0rc4 - VT02.2

- Made isolated `venv` plus pip the primary installation route.
- Added SciPy to the base runtime metadata and a validated `cuda12` CuPy extra.
- Repaired the optional Conda environment and added pip-first installation checks.

## 0.1.0rc3 - VT02.1

- Clarified analysis-coordinate and default-export metadata without changing scientific outputs.
- Removed stale release identity and legacy package wording from the user-facing release.

## 0.1.0rc2 - VT02

- Added self-sufficient, hash-validated source provenance to compact alignment results.
- Added `mdcorealign apply RESULT_DIR -o OUTPUT` with suffix-driven PDB, mmCIF and XTC export.
- Added default complete-protein export using saved C-alpha-derived transforms.
- Added core-label occupancy and robust-RMSF B-factor annotations for PDB/mmCIF.
- Added the complete residue-profile mmCIF category and XTC annotated topology companions.
- Preserved all scientific algorithms and kept CA-only input CA-only with an explicit warning.

## 0.1.0rc1 - Unreleased

Initial release candidate for user testing.

- Added `mdcorealign` package namespace and console command.
- Added robust alignment, flexibility profiling, diagnostics, and preprocessing audit CLI commands.
- Added toy CA-only multi-model PDB quickstart.
- Added user-facing installation, CLI, API, output-format, preprocessing, troubleshooting, release, and user-testing documentation.
- Added packaging scripts, CI workflow, license, citation metadata, and release checklist.
- Added explicit `align --mode exact` and `align --mode throughput` product aliases.
- Added stable transform, frame-metric, residue-profile, segment, and diagnostics outputs.
- Added `mdcorealign apply` for deferred coordinate materialisation.
- Added CPU/GPU throughput, exact, local-deformation, and apply examples.

<!-- BEGIN MDCOREALIGN_RELEASE_METADATA -->
## Release-Candidate Metadata

- Current delivery candidate: `0.1.0rc6` (publication status recorded separately)
- Repository: `pending`
- Zenodo DOI: `pending`
- TestPyPI URL: `pending`
- This release candidate includes package installation, CLI, toy quickstart, clean-install validation, build/twine checks, and manuscript-readiness audit artifacts.
<!-- END MDCOREALIGN_RELEASE_METADATA -->
