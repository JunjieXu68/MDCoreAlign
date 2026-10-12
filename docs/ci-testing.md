# CI testing boundaries

CI keeps installed-product tests, source research tests, and historical-result
checks separate. The classification and its reasons live in
`tests/test_layer_manifest.json`; pytest accepts
`--test-layer=product|research|archive|all`, with `product` as its default.
Changing a test's layer must preserve its coverage and document why the layer
needs source-only modules or historical data.

The existing Linux CPU matrix retains Python 3.11, 3.12 and 3.13 with
`fail-fast: false`. Each interpreter installs `requirements-ci-product.txt`
for product tests and `requirements-ci-quality.txt` for build/lint/type checks.
Research tests additionally install `requirements-ci-research.txt`. These files
are CI dependencies, not new ordinary-user requirements or changes to rc8's
package metadata. In particular, setuptools is an explicit test dependency;
an isolated build dependency does not install it into the test interpreter.

## Installed product

Build a wheel, install that wheel normally into the test interpreter, and install
the CI test dependencies. Do not use an editable install. For a new CPU virtual
environment, from the full checkout:

```bash
python -m venv /tmp/mdcorealign-ci-env-NEW
source /tmp/mdcorealign-ci-env-NEW/bin/activate
python -m pip install -r requirements-ci-product.txt -r requirements-ci-quality.txt
python -m build
python -m twine check dist/*
python -m pip install dist/mdcorealign-0.1.0rc8-py3-none-any.whl
python -m pip check
```

Then run the selected wheel's tests, supplying its real hashes:

```bash
python tools/ci/run_tests.py --test-layer product \
  --artifact-kind candidate --expected-version 0.1.0rc8 \
  --wheel /absolute/path/mdcorealign-0.1.0rc8-py3-none-any.whl \
  --expected-wheel-sha256 ACTUAL_SHA256 \
  --sdist /absolute/path/mdcorealign-0.1.0rc8.tar.gz \
  --expected-sdist-sha256 ACTUAL_SDIST_SHA256 \
  --output-dir /tmp/mdcorealign-product-NEW
```

`candidate` identifies the wheel built from the current CI checkout. To validate
the already published rc8 release, use `--artifact-kind release` and the actual
independently downloaded release wheel and its accepted SHA256. The runner does
not download, build, install or silently replace packages. A published rc8 wheel
is not evidence that a newly changed runtime implementation works.

The optional paired `--sdist` and `--expected-sdist-sha256` arguments select the
actual source archive for packaging contract tests. Both artifact paths are
forwarded through their explicit `MDCOREALIGN_*_UNDER_TEST` variables; inherited
artifact variables are cleared.

The runner uses an external temporary working directory and isolated Python,
removes `PYTHONPATH` and user-site imports, and rejects editable installation,
checkout `src` injection, or product imports outside this interpreter's
`site-packages`. It verifies the distribution version, imported version, and
installed runtime bytes against the selected wheel before and after collection
and execution. Every test also checks loaded product module origins and bytes.
Tests may load test/helper code from the checkout; this does not permit importing
product runtime code from `src`. Subprocess helpers inherit the explicit product
layer and must preserve that boundary.

The runner first collects the selected product tests, then executes exactly the
same item set. Both phases have separate logs, JSON identity/count reports and
JUnit XML. A collection-only pass is never reported as executed tests. Native
GPU tests retain their explicit GPU availability gates; this CPU CI setup does
not claim real-GPU validation.

## Source research/development

Source-only research modules remain in the full Git checkout rather than the
product wheel. Run their software tests separately:

```bash
python -m pip install -r requirements-ci-research.txt
python tools/ci/run_tests.py --test-layer research \
  --artifact-kind source --output-dir /tmp/mdcorealign-research-NEW
```

Only this layer and explicit historical archive work permit
`PYTHONPATH=/absolute/checkout/src`. The runner checks that product modules used
in this layer originate in that source tree. The report says `source`; it is not
an installed-wheel acceptance report. This layer is still software testing, not
a request to launch a scientific benchmark or simulation.

## Historical archives

The exact historical nodes and required artifacts are recorded in
`tests/archive_qa_nodes.json`. Default CI checks that these tests can be collected
separately, without pretending that an empty checkout contains preserved results:

```bash
python tools/ci/run_tests.py --test-layer archive \
  --artifact-kind source --output-dir /tmp/mdcorealign-archive-collection-NEW
```

This produces `COLLECTED_ONLY`, not `PASS` for historical assertions. To execute
them where the original artifact paths are available, add `--run-archive` and
choose a new output directory. The runner resolves literal paths and glob patterns in that manifest. If any
declared required fixture is missing, explicit archive execution stops with a
nonzero exit after successful collection, even if an original test would skip
on missing input. Additional original assertions remain unchanged. Existing
historical evidence is not regenerated by this runner.

Reports distinguish collected and deselected items, unique passed/skipped/failed
items, setup/teardown errors, and raw pytest reports. Raw report counts include
test phases and any unittest subtest reports; do not add overlapping counts.
Each run records the interpreter, working directory, layer, candidate/release
identity, wheel hash and runner/guard hashes. It also records the orchestration
checkout Git HEAD/tree, dirty status, tracked diff hash, and separate untracked
file hashes. `--artifact-source-commit` may declare the source commit used to
build the selected wheel. That release/build identity is separate from the newer
CI/test orchestration commit; a dirty local checkout is reported explicitly. Output directories must be new so
failed attempts remain available for inspection.
