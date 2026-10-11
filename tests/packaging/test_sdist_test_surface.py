"""Validate the test surface actually shipped with the stable source distribution."""

from __future__ import annotations

import ast
import os
import tarfile
from collections import deque
from pathlib import Path

from setuptools._distutils.filelist import FileList

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_MODULES = {
    "mdcorealign.benchmark_minimal",
    "mdcorealign.experimental_gpu_batch",
    "mdcorealign.experimental_stream",
    "mdcorealign.experimental_throughput_v2",
    "mdcorealign.guarded",
}
SOURCE_ONLY_TESTS = {
    "tests/test_alignment.py",
    "tests/test_soft_superposition.py",
    "tests/test_synthetic_pipeline.py",
    "tests/guarded_fast_global_v1/test_guarded_cascade.py",
    "tests/no_materialisation_v1/test_benchmark_minimal.py",
    "tests/post_b3_optimisation_v1/test_experimental_gpu_batch.py",
    "tests/stream_align_v1/test_stream_alignment.py",
    "tests/throughput_v2_sparse_keyframe/test_experimental_throughput_v2.py",
    "tests/analysis/test_core_disagreement_sensitivity_sampling_v1.py",
    "tests/analysis/test_biological_core_validation_v1.py",
    "tests/analysis/test_core_disagreement_rmsf_v1.py",
    "tests/analysis/test_core_disagreement_sensitivity_v1.py",
    "tests/test_atlas_strict_single_thread_cpu_v1.py",
    "tests/test_atlas_untimed_export_recovery.py",
    "tests/packaging/test_build_surface.py",
}


def _files_under_test(record_property):
    artifact = os.environ.get("MDCOREALIGN_SDIST_UNDER_TEST")
    if artifact:
        path = Path(artifact).resolve()
        record_property("sdist_policy_scope", "actual_sdist_archive")
        record_property("sdist_artifact", str(path))
        with tarfile.open(path) as archive:
            return {
                member.name.split("/", 1)[1]: archive.extractfile(member).read().decode("utf-8")
                for member in archive.getmembers()
                if member.isfile() and member.name.endswith(".py")
            }
    # Fast full-checkout regression: expand real MANIFEST commands using the
    # same setuptools file-list engine, including ordering of include/exclude.
    # Artifact acceptance sets the environment above; neither branch skips.
    all_files = [
        path.relative_to(ROOT).as_posix()
        for directory in ("src", "tests", "scripts", "examples", "docs")
        for path in (ROOT / directory).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    all_files += [path.name for path in ROOT.iterdir() if path.is_file()]
    manifest = FileList()
    manifest.allfiles = all_files
    manifest.files = [
        name
        for name in all_files
        if name.startswith(("src/mdcorealign/", "src/mdsuperpose/")) and name.endswith(".py")
    ]
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            manifest.process_template_line(line)
    record_property("sdist_policy_scope", "setuptools_manifest_expansion")
    return {name: (ROOT / name).read_text() for name in set(manifest.files) if name.endswith(".py")}


def _module_name(path):
    relative = path.removeprefix("src/")
    return relative[:-3].replace("/", ".").removesuffix(".__init__")


def _imports(path, text):
    dependencies = []
    for node in ast.walk(ast.parse(text, filename=path)):
        if isinstance(node, ast.Import):
            dependencies.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                module = _module_name(path).split(".")
                if path.endswith("/__init__.py"):
                    module.append("__init__")
                base = ".".join(module[: -node.level] + ([base] if base else []))
            dependencies.extend([base, *(base + "." + alias.name for alias in node.names)])
        elif isinstance(node, ast.Call) and node.args:
            # Include literal dynamic imports. Arbitrary executable import
            # construction is also covered by the separate actual collection gate.
            function = node.func
            if (isinstance(function, ast.Name) and function.id == "__import__") or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            ):
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    dependencies.append(value.value)
    return dependencies


def _unavailable(module):
    return (
        module == "mdstablecore"
        or module.startswith("mdstablecore.")
        or any(module == banned or module.startswith(banned + ".") for banned in EXCLUDED_MODULES)
    )


def test_sdist_retains_public_tests_and_omits_only_named_source_tests(record_property):
    files = _files_under_test(record_property)
    assert not SOURCE_ONLY_TESTS.intersection(files)
    assert {
        "tests/fix_and_release/test_parameter_contracts.py",
        "tests/fix_and_release/test_input_export_contracts.py",
        "tests/m8_delivery/test_public_update_execution.py",
        "tests/test_product_contract_cli.py",
        "tests/advisor_release_blocker_cleanup_v3/test_release_contract_v3.py",
        "tests/packaging/test_sdist_test_surface.py",
    } <= files.keys()


def test_sdist_shipped_tests_do_not_reach_excluded_modules(record_property):
    files = _files_under_test(record_property)
    modules = {_module_name(path): path for path in files}
    dependencies = {path: _imports(path, text) for path, text in files.items()}
    failures = []
    for test in sorted(path for path in files if path.startswith("tests/")):
        pending = deque([(test, [])])
        seen = set()
        while pending:
            path, chain = pending.popleft()
            if path in seen:
                continue
            seen.add(path)
            for module in dependencies[path]:
                if _unavailable(module):
                    failures.append(" -> ".join([*chain, path, module]))
                elif module in modules:
                    pending.append((modules[module], [*chain, path]))
    assert not failures, "Unshipped module dependencies:\n" + "\n".join(failures)
