# Release process and identity

This source tree identifies MDCoreAlign as version `1.0.0`. Its numerical
implementation, default scientific parameters and output schema are unchanged
from the maintained rc8 runtime. The version update and installation guidance
do not establish that historical scientific analyses were rerun with 1.0.0.
Historical VT02.2 (`0.1.0rc4`) and later candidate artifacts retain their original
identities: VT02.2 predates current M8 support, and rc5 predates the interface
fixes delivered in rc6.

The official repository is
[github.com/JunjieXu68/MDCoreAlign](https://github.com/JunjieXu68/MDCoreAlign).
The [v1.0.0 Release](https://github.com/JunjieXu68/MDCoreAlign/releases/tag/v1.0.0)
is the publication record for the tag, associated source commit, downloadable
assets and validation evidence. Verify that record and its asset hashes before
attributing a local package to the release. This document specifies the release
checks; it does not assert that an unpublished build or unfinished check passed.

## Build and validate

Build the direct wheel and sdist from a clean checkout of the full source commit
being released. Extract that sdist into another clean directory and build its
wheel. Compare runtime module names and bytes, package metadata and entry points.
The five excluded research modules remain in the full development source tree,
outside the installed runtime surface.

Install the selected artifacts into independent environments. Invoke the
installed CLI from outside the source tree, without source `PYTHONPATH` or
user-site imports. Run the installed-product suite against the explicit wheel
and sdist, followed by the separate source-research tests. Verify actual update
counts and delta values, input replay and coordinate export. Report unavailable
backends explicitly; CPU CI does not establish real-GPU validation. Run shipped
examples from the matching source delivery. See
[CI testing boundaries](ci-testing.md) for commands, package identity checks and
the separate historical-artifact tier.

Before a new tag exists remotely, validate the proposed local source commit and
its built artifacts. After publication, check installation from the actual tag
and the downloaded release assets. Preserve failures and distinguish the local
pre-publication checks from this remote verification.

## Publish and archive

Preserve remote history and existing release assets. Review staged files for
credentials, private documents and excluded data before publishing. After the
push, tag and release, verify the remote commit and tag target, release flags
and downloaded asset hashes. A Git commit, package SHA-256 and Zenodo DOI identify
different objects; none is a substitute for the others.

Use the assigned Zenodo release record when it becomes available and then record
its actual DOI. No DOI is invented or inferred from repository integration being
enabled. PyPI/TestPyPI publication is not claimed by this release workflow; the
[README](../README.md) documents fixed-tag GitHub installation.

## Historical scientific evidence

Archived scientific tables, figures and timings keep their original code,
input, parameter and execution provenance. A software test pass or a 1.0.0
release identity does not reproduce those results. Source-research tests check
helper implementations; historical-artifact QA checks existing artifacts and
requires their declared fixtures. Neither scope silently launches a benchmark
or regenerates historical evidence.

<!-- BEGIN MDCOREALIGN_RELEASE_METADATA -->
## Current Release Metadata

- Version: `1.0.0`
- Release date: `2026-10-12`
- Repository: <https://github.com/JunjieXu68/MDCoreAlign>
- Tag, commit and assets: <https://github.com/JunjieXu68/MDCoreAlign/releases/tag/v1.0.0>
- Zenodo DOI: see the linked GitHub Release for the assigned archival DOI
- PyPI/TestPyPI publication: not claimed
<!-- END MDCOREALIGN_RELEASE_METADATA -->
