# Candidate Release Checklist

- [ ] Confirm the authorized repository, branch, visibility and permissions.
- [ ] Preserve existing remote history and old candidate artifacts.
- [ ] Review staged files and all pending history for secrets and private data.
- [ ] Record the full source commit; build from its clean checkout.
- [ ] Pass portable software tests; report archive QA and GPU availability separately.
- [ ] Compare direct and sdist-derived runtime modules, metadata and entry points.
- [ ] Pass installed CLI/API tests outside the source tree for each build route.
- [ ] Trace actual M4/M8 updates and default/explicit delta on available backends.
- [ ] Verify example fixtures, README quickstart, align and coordinate export.
- [ ] Run pip check, doctor, help and package metadata checks.
- [ ] Record dependency backup payloads and actual offline restoration scope.
- [ ] Publish the authorized candidate tag and GitHub prerelease if all gates pass.
- [ ] Verify remote commit/tag, release flags and downloaded asset SHA256 values.
- [ ] Record actual status and limitations in RELEASE_READINESS.md.

PyPI/TestPyPI, DOI registration, full streaming, timing redesign and all-platform
validation are outside this candidate task. A missing publication target must
not prevent local fixes, testing, commit and artifact delivery.
