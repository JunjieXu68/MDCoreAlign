"""Identity checks and machine-readable counts for the three CI test layers."""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import sysconfig
import zipfile
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IdentityPlugin:
    """Fail on a source/installed identity violation instead of skipping tests."""

    def __init__(self, context: dict):
        self.context = context
        self.layer = context['test_layer']
        self.checkout = Path(context['checkout']).resolve()
        self.src = self.checkout / 'src'
        self.prefix = Path(sys.prefix).resolve()
        self.sites = {Path(sysconfig.get_paths()[key]).resolve() for key in ('purelib', 'platlib')}
        self.checks: list[dict] = []
        self.nodeids: list[str] = []
        self.deselected: list[str] = []
        self.reports: list[dict] = []
        self.collection_errors: list[dict] = []
        self.expected_payload: dict[str, bytes] = {}
        self.site_root: Path | None = None
        if self.layer == 'product':
            wheel = Path(context['wheel'])
            if sha(wheel) != context['wheel_sha256']:
                raise RuntimeError('Wheel bytes changed before pytest startup')
            with zipfile.ZipFile(wheel) as archive:
                self.expected_payload = {
                    name: archive.read(name) for name in archive.namelist()
                    if name.endswith('.py') and name.startswith(('mdcorealign/', 'mdsuperpose/'))
                }
            if not self.expected_payload:
                raise RuntimeError('Wheel has no product Python modules')

    def check(self, stage: str, *, all_files: bool = False) -> None:
        problems = []
        origins = {}
        if self.layer == 'product':
            if os.environ.get('PYTHONPATH'):
                problems.append('PYTHONPATH is set during installed-product tests')
            if self.src in {Path(p).resolve() for p in sys.path if p}:
                problems.append('Checkout src was inserted into sys.path during product tests')
            distribution = importlib.metadata.distribution('mdcorealign')
            direct = distribution.read_text('direct_url.json')
            if direct and json.loads(direct).get('dir_info', {}).get('editable'):
                problems.append('Editable distribution is forbidden in product tests')
            if distribution.version != self.context['expected_version']:
                problems.append('Installed distribution version differs from expected version')
        elif os.environ.get('PYTHONPATH') != str(self.src):
            problems.append('Source layers require exactly checkout/src as PYTHONPATH')
        for name, module in list(sys.modules.items()):
            root = name.split('.', 1)[0]
            if root not in {'mdcorealign', 'mdsuperpose'}:
                continue
            filename = getattr(module, '__file__', None)
            if not filename:
                continue
            path = Path(filename).resolve()
            origins[name] = str(path)
            if self.layer == 'product':
                sites = [site for site in self.sites if path.is_relative_to(site)]
                if not sites or not path.is_relative_to(self.prefix):
                    problems.append(f'{name} imported outside this interpreter site-packages: {path}')
                    continue
                relative = path.relative_to(sites[0]).as_posix()
                expected = self.expected_payload.get(relative)
                if expected is None or path.read_bytes() != expected:
                    problems.append(f'{name} bytes do not match the selected wheel: {path}')
            elif not path.is_relative_to(self.src):
                problems.append(f'{name} imported outside source-layer checkout/src: {path}')
        if self.layer == 'product' and all_files:
            for name, expected in self.expected_payload.items():
                matches = [site / name for site in self.sites if (site / name).is_file()]
                if not matches or any(not p.is_relative_to(self.prefix) or p.read_bytes() != expected for p in matches):
                    problems.append(f'Installed module missing or byte mismatch: {name}')
            version = getattr(sys.modules.get('mdcorealign'), '__version__', None)
            if version != self.context['expected_version']:
                problems.append('Imported mdcorealign.__version__ differs from expected version')
        self.checks.append({'stage': stage, 'status': 'FAIL' if problems else 'PASS',
                            'problems': problems, 'loaded_product_module_count': len(origins),
                            'origins': origins if all_files or problems else None})
        if problems:
            raise RuntimeError('CI package identity violation:\n' + '\n'.join(problems))

    def pytest_sessionstart(self, session) -> None:
        del session
        # Establish and record what this interpreter actually imports before
        # pytest imports test modules; later hooks reject any path contamination.
        importlib.import_module('mdcorealign')
        importlib.import_module('mdsuperpose')
        self.check('before_collection', all_files=True)

    def pytest_collection_finish(self, session) -> None:
        self.nodeids = [item.nodeid for item in session.items]
        self.check('after_collection', all_files=True)

    def pytest_deselected(self, items) -> None:
        self.deselected.extend(item.nodeid for item in items)

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.collection_errors.append({'nodeid': report.nodeid, 'message': str(report.longrepr)})

    def pytest_runtest_setup(self, item) -> None:
        self.check('before_test:' + item.nodeid)

    def pytest_runtest_teardown(self, item) -> None:
        self.check('after_test:' + item.nodeid)

    def pytest_runtest_logreport(self, report) -> None:
        row = {'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome}
        if report.skipped or report.failed:
            row['longrepr'] = str(report.longrepr)
        if getattr(report, 'wasxfail', None):
            row['wasxfail'] = report.wasxfail
        self.reports.append(row)

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        del session, exitstatus
        self.check('session_finish', all_files=True)

    def write_report(self, exitcode: int) -> None:
        states: dict[str, str] = {}
        precedence = {'passed': 0, 'skipped': 1, 'failed': 2, 'error': 3}
        for row in self.reports:
            if row['when'] != 'call' and row['outcome'] == 'passed':
                continue
            state = 'error' if row['outcome'] == 'failed' and row['when'] != 'call' else row['outcome']
            if precedence[state] >= precedence.get(states.get(row['nodeid'], ''), -1):
                states[row['nodeid']] = state
        counts = {state: sum(value == state for value in states.values())
                  for state in ('passed', 'skipped', 'failed', 'error')}
        value = {'test_layer': self.layer, 'phase': self.context['phase'], 'exit_code': exitcode,
                 'artifact_kind': self.context['artifact_kind'], 'python': sys.executable,
                 'prefix': str(self.prefix), 'cwd': str(Path.cwd()), 'sys_path': sys.path,
                 'collected': len(self.nodeids), 'nodeids': self.nodeids,
                 'deselected': len(self.deselected), 'deselected_nodeids': self.deselected,
                 'collection_errors': self.collection_errors, 'unique_item_outcomes': counts,
                 'report_count': len(self.reports), 'reports': self.reports,
                 'identity_checks': self.checks,
                 'count_policy': 'Unique pytest item outcomes; setup/teardown failures are errors. '
                                 'Raw report count includes phases and any unittest subtest reports; do not sum scopes.'}
        Path(self.context['phase_report']).write_text(json.dumps(value, indent=2) + '\n')
