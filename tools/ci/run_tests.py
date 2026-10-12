#!/usr/bin/env python3
"""Collect and run explicit installed-product, source-research or archive layers."""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[2]
PLUGIN = Path(__file__).with_name('pytest_identity.py')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + '\n')


def orchestration_identity() -> dict:
    """Record the checkout running tests, separately from the wheel under test."""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

    def git(*arguments: str) -> bytes:
        return subprocess.run(['git', *arguments], cwd=CHECKOUT, env=env,
                              capture_output=True, check=True).stdout

    status = git('status', '--porcelain=v1', '--untracked-files=all')
    diff = git('diff', '--binary', 'HEAD')
    untracked = git('ls-files', '--others', '--exclude-standard', '-z')
    files = []
    for raw in untracked.split(b'\0'):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        path = CHECKOUT / relative
        files.append({'path': relative, 'sha256': sha(path) if path.is_file() else None})
    return {'head_commit': git('rev-parse', 'HEAD').decode().strip(),
            'head_tree': git('rev-parse', 'HEAD^{tree}').decode().strip(),
            'dirty': bool(status), 'status_porcelain': status.decode(errors='replace'),
            'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
            'untracked_files': files, 'github_sha': os.environ.get('GITHUB_SHA'),
            'scope': 'Test orchestration checkout identity; tracked diff hash excludes separately listed untracked files. '
                     'This does not change or identify the independently selected release wheel.'}


def archive_fixtures() -> dict:
    manifest = CHECKOUT / 'tests/archive_qa_nodes.json'
    requirements = json.loads(manifest.read_text())['tests']
    locations = sorted({p for row in requirements for p in row.get('fixture_locations', [])})
    records = []
    for relative in locations:
        path = CHECKOUT / relative
        matches = sorted(glob.glob(str(path), recursive=True)) if glob.has_magic(str(path)) else ([str(path)] if path.exists() else [])
        records.append({'path': str(path), 'resolution': 'glob' if glob.has_magic(str(path)) else 'literal',
                        'exists': bool(matches), 'matched_paths': matches})
    return {'fixture_manifest': str(manifest), 'fixture_manifest_sha256': sha(manifest),
            'fixtures': records, 'missing_fixture_count': sum(not row['exists'] for row in records),
            'fixture_policy': 'Collection is not execution. Explicit archive execution is blocked if any declared '
                              'required path/glob has no match; preserved assertions may have additional requirements.'}


def child(context_path: Path) -> int:
    context = json.loads(context_path.read_text())
    spec = importlib.util.spec_from_file_location('mdcorealign_ci_identity', PLUGIN)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load CI identity plugin')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    guard = module.IdentityPlugin(context)
    import pytest
    code = 3
    try:
        code = int(pytest.main(context['pytest_args'], plugins=[guard]))
        return code
    finally:
        guard.write_report(code)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-layer', choices=('product', 'research', 'archive'), default='product')
    parser.add_argument('--artifact-kind', choices=('candidate', 'release', 'source'))
    parser.add_argument('--expected-version')
    parser.add_argument('--artifact-source-commit', help='Declared build/release source commit, separate from orchestration Git HEAD')
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--expected-wheel-sha256')
    parser.add_argument('--sdist', type=Path, help='Actual sdist for packaging contract tests')
    parser.add_argument('--expected-sdist-sha256')
    parser.add_argument('--output-dir', type=Path, required=True, help='New evidence directory; must not already exist')
    parser.add_argument('--run-archive', action='store_true', help='Execute original historical assertions after collecting; fixtures must exist')
    args = parser.parse_args()
    if args.test_layer == 'product':
        if args.artifact_kind not in ('candidate', 'release') or not args.wheel or not args.expected_wheel_sha256 or not args.expected_version:
            parser.error('product requires --artifact-kind candidate|release, --wheel, --expected-wheel-sha256 and --expected-version')
        args.wheel = args.wheel.resolve()
        if sha(args.wheel) != args.expected_wheel_sha256:
            parser.error('selected wheel SHA256 does not match expected SHA256')
    elif args.artifact_kind not in (None, 'source'):
        parser.error('research/archive layers use --artifact-kind source, never installed-product claims')
    if bool(args.sdist) != bool(args.expected_sdist_sha256):
        parser.error('--sdist and --expected-sdist-sha256 must be supplied together')
    if args.sdist:
        args.sdist = args.sdist.resolve()
        if sha(args.sdist) != args.expected_sdist_sha256:
            parser.error('selected sdist SHA256 does not match expected SHA256')
    if args.run_archive and args.test_layer != 'archive':
        parser.error('--run-archive applies only to the archive layer')
    version = args.expected_version or tomllib.loads((CHECKOUT / 'pyproject.toml').read_text())['project']['version']
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cwd = Path(tempfile.mkdtemp(prefix='mdcorealign-ci-' + args.test_layer + '-')).resolve()
    if cwd.is_relative_to(CHECKOUT):
        raise RuntimeError('CI working directory must be outside the checkout')
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PYTEST_ADDOPTS', 'PYTEST_PLUGINS'):
        env.pop(key, None)
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env.update({'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'MDCOREALIGN_TEST_LAYER': args.test_layer,
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'NUMEXPR_NUM_THREADS': '1'})
    if args.test_layer != 'product':
        env['PYTHONPATH'] = str(CHECKOUT / 'src')
    for name in ('MDCOREALIGN_WHEEL_UNDER_TEST', 'MDCOREALIGN_SDIST_UNDER_TEST'):
        env.pop(name, None)
    if args.wheel:
        env['MDCOREALIGN_WHEEL_UNDER_TEST'] = str(args.wheel)
    if args.sdist:
        env['MDCOREALIGN_SDIST_UNDER_TEST'] = str(args.sdist)
    context = {'test_layer': args.test_layer, 'artifact_kind': args.artifact_kind or 'source',
               'checkout': str(CHECKOUT), 'expected_version': version,
               'orchestration_git': orchestration_identity(),
               'artifact_source_commit_declared': args.artifact_source_commit,
               'wheel': str(args.wheel) if args.wheel else None,
               'wheel_sha256': args.expected_wheel_sha256, 'python': sys.executable,
               'sdist': str(args.sdist) if args.sdist else None, 'sdist_sha256': args.expected_sdist_sha256,
               'cwd': str(cwd), 'executable_search_prefix': str(Path(sys.executable).parent), 'runner_sha256': sha(Path(__file__)), 'guard_sha256': sha(PLUGIN),
               'start_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'invocation.json', context)
    phases = ['collect'] if args.test_layer == 'archive' and not args.run_archive else ['collect', 'test']
    summary = dict(context, status='RUNNING', phases=[], archive_execution_requested=args.run_archive)
    if args.test_layer == 'archive':
        summary.update(archive_fixtures())
    for phase in phases:
        if phase == 'test' and args.test_layer == 'archive' and summary['missing_fixture_count']:
            summary.update(status='FAIL', execution_blocked_missing_fixtures=True,
                           reason='Archive collection succeeded, but declared historical fixtures are missing. No archive assertions executed.')
            write_json(output / 'summary.json', summary)
            print(summary['reason'], file=sys.stderr, flush=True)
            return 2
        phase_report = output / (phase + '_report.json')
        pytest_args = ['-q', '--import-mode=importlib', '--test-layer=' + args.test_layer,
                       '--rootdir=' + str(CHECKOUT), '-c', str(CHECKOUT / 'pyproject.toml'),
                       '--junitxml=' + str(output / (phase + '.junit.xml')), str(CHECKOUT / 'tests')]
        if phase == 'collect':
            pytest_args.insert(0, '--collect-only')
        phase_context = dict(context, phase=phase, phase_report=str(phase_report), pytest_args=pytest_args)
        context_path = output / (phase + '_context.json')
        write_json(context_path, phase_context)
        argv = [sys.executable, '-B']
        if args.test_layer == 'product':
            argv.append('-I')
        argv += [str(Path(__file__).resolve()), '_pytest', str(context_path)]
        with (output / (phase + '.stdout.log')).open('w') as stdout, (output / (phase + '.stderr.log')).open('w') as stderr:
            result = subprocess.run(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False)
        details = json.loads(phase_report.read_text()) if phase_report.exists() else {}
        row = {'phase': phase, 'argv': argv, 'returncode': result.returncode,
               'collected': details.get('collected'), 'unique_item_outcomes': details.get('unique_item_outcomes'),
               'report': str(phase_report), 'junit': str(output / (phase + '.junit.xml'))}
        summary['phases'].append(row)
        print(json.dumps(row), flush=True)
        if result.returncode != 0 or not details.get('collected'):
            summary['status'] = 'FAIL'
            write_json(output / 'summary.json', summary)
            return result.returncode or 1
        if phase == 'test':
            collection = json.loads((output / 'collect_report.json').read_text())
            if sorted(details['nodeids']) != sorted(collection['nodeids']):
                summary.update(status='FAIL', reason='Actual test selection differs from collection-only selection')
                write_json(output / 'summary.json', summary)
                return 1
    summary.update(status='COLLECTED_ONLY' if phases == ['collect'] else 'PASS',
                   end_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'output_dir': str(output)}), flush=True)
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '_pytest':
        raise SystemExit(child(Path(sys.argv[2])))
    raise SystemExit(main())
