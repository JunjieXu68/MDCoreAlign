# Release Candidate Process

The current candidate is `0.1.0rc7`. It includes parameter and input/export
contract fixes while retaining default M4 and actual M8 execution. It is not a
final stable release. Historical VT02.2 / `0.1.0rc4` and the distinct M8-only
`0.1.0rc5` artifacts must remain unchanged. VT02.2 does not include the current
M8 support; rc5 does not include the interface fixes delivered in rc6.

Build the final direct wheel and sdist from a clean checkout of the full source
commit named in the acceptance record. Extract that sdist to another clean
directory and build its wheel. Compare runtime module names and bytes, package
metadata and entry points. The five excluded research modules remain in the
development source tree, outside the installed runtime surface.

Run the portable software suite, then install each build route into independent
environments. Invoke the installed CLI from outside the source tree, without
source `PYTHONPATH` or user-site imports. Verify actual update counts and delta
values in CPU/GPU kernels, input replay and coordinate export. Report unavailable
backends explicitly. Run the shipped examples from the matching source delivery.
See [testing.md](testing.md) for the separate archive QA tier.

The accompanying `RELEASE_READINESS.md` is the authoritative delivery record:
it lists full Git commit, candidate tag if created, wheel/sdist SHA256, exact
test scopes, limitations and publication status. File hashes are not Git
commits. A local commit and local acceptance are not a GitHub publication.

Publish only to a confirmed repository with known visibility and adequate
permission. Preserve existing remote history. Review the exact staged files
and pending history for credentials, private documents and excluded data.
After push/tag/prerelease, verify the remote commit and tag target, release
flags and downloaded asset hashes. A fresh-clone/install check establishes
technical retrieval, not another person's access to a private repository.

<!-- BEGIN MDCOREALIGN_RELEASE_METADATA -->
## Release Metadata Status

- Candidate: `0.1.0rc7`
- Repository, full commit, tag and publication: see `RELEASE_READINESS.md`
- Zenodo DOI: pending
- PyPI/TestPyPI publication: not part of this candidate task

<!-- END MDCOREALIGN_RELEASE_METADATA -->
