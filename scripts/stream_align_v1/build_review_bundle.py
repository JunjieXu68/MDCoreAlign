#!/usr/bin/env python3
"""Build and independently validate the mandatory stream_align_v1 handoff."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from urllib.parse import urlparse

from scripts.stream_align_v1.common import OUT, ROOT, read_tsv, write_json, write_text, write_tsv


DATE = datetime.now().astimezone().strftime("%Y%m%d")
ARCHIVE = OUT / f"stream_align_v1_review_bundle_{DATE}.tar.gz"
CHECKSUM = ARCHIVE.with_suffix(ARCHIVE.suffix + ".sha256")
VALIDATION = OUT / f"stream_align_v1_review_bundle_{DATE}_validation.json"
TEXT_SUFFIXES = {
    "",
    ".csv",
    ".diff",
    ".json",
    ".log",
    ".md",
    ".patch",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sanitize(text: str) -> str:
    replacements = (
        (str(ROOT) + "/", ""),
        (str(ROOT), "."),
        ("/home/jjxu/", "<USER_HOME>/"),
        ("/storage/junjie/", "<STORAGE_ROOT>/"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    text = re.sub(r"/tmp/[A-Za-z0-9._/-]+", "<TEMP_PATH>", text)
    return text


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        try:
            destination.write_text(_sanitize(source.read_text()))
            return
        except UnicodeDecodeError:
            pass
    shutil.copy2(source, destination)


def _required() -> list[str]:
    return [
        "00_README_FIRST.md",
        "final_decision_report.md",
        "phase_status.tsv",
        "key_metrics.tsv",
        "file_manifest.tsv",
        "sha256_manifest.tsv",
        "algorithm_design.md",
        "complexity_analysis.md",
        "landmark_ablation.tsv",
        "keyframe_ablation.tsv",
        "refinement_ablation.tsv",
        "atlas_quality_speed.tsv",
        "long_scaling_results.tsv",
        "quality_runtime_pareto.tsv",
        "failure_warning_atlas.tsv",
        "figures/fig_s2_landmark_pareto.pdf",
        "figures/fig_s4_keyframe_ablation.pdf",
        "figures/fig_s6_atlas_quality_speed.pdf",
        "figures/fig_s6_synthetic_scaling.pdf",
        "source/src/mdcorealign/experimental_stream.py",
        "source/scripts/stream_align_v1/run_stream_benchmarks.py",
        "source/tests/stream_align_v1/test_stream_alignment.py",
        "source/unified_source_diff.patch",
        "validation/focused_test_output.log",
        "validation/full_test_output.log",
        "validation/frozen_integrity_report.md",
    ]


def _stage_outputs(bundle: Path) -> None:
    root_files = (
        "final_decision_report.md",
        "phase_status.tsv",
        "key_metrics.tsv",
        "algorithm_design.md",
        "complexity_analysis.md",
        "landmark_ablation.tsv",
        "keyframe_ablation.tsv",
        "refinement_ablation.tsv",
        "atlas_quality_speed.tsv",
        "long_scaling_results.tsv",
        "quality_runtime_pareto.tsv",
        "failure_warning_atlas.tsv",
        "tail_panel_results.tsv",
        "s7_decision.json",
        "s6_execution_status.json",
    )
    for name in root_files:
        _copy(OUT / name, bundle / name)
    for directory in (
        "figures",
        "frozen_landmarks",
        "interval_audits",
        "s0_freeze",
        "s1_contracts",
        "development_diagnostics",
    ):
        for source in sorted((OUT / directory).rglob("*")):
            if source.is_file():
                _copy(source, bundle / source.relative_to(OUT))
    for source in sorted((OUT / "validation").glob("*")):
        if source.is_file():
            _copy(source, bundle / "validation" / source.name)


def _source_files() -> list[str]:
    return [
        "src/mdcorealign/experimental_stream.py",
        "scripts/stream_align_v1/common.py",
        "scripts/stream_align_v1/run_s0_s1_registry.py",
        "scripts/stream_align_v1/run_stream_benchmarks.py",
        "scripts/stream_align_v1/finalize_s7.py",
        "scripts/stream_align_v1/verify_frozen_integrity.py",
        "tests/stream_align_v1/test_stream_alignment.py",
    ]


def _stage_source(bundle: Path) -> None:
    for relative in _source_files():
        _copy(ROOT / relative, bundle / "source" / relative)
    patches: list[str] = []
    for relative in _source_files():
        lines = _sanitize((ROOT / relative).read_text()).splitlines(keepends=True)
        patches.extend(
            difflib.unified_diff(
                [],
                lines,
                fromfile="/dev/null",
                tofile=f"b/{relative}",
            )
        )
    write_text(bundle / "source/unified_source_diff.patch", "".join(patches))


def _entry(bundle: Path) -> None:
    write_text(
        bundle / "00_README_FIRST.md",
        """# stream_align_v1 Review Handoff

This archive is the complete S0-S8 evidence package for an independent,
experimental long-trajectory branch. No old baseline or external comparator
was rerun. Start with `final_decision_report.md` and `key_metrics.tsv`.

The locked result is negative: neither balanced nor fast stream passed. Pooled
ATLAS medians were favourable, but 3nci_A and 5vz0_A failed the fast quality
contract, all 13 locked difficult cases remained unresolved, and the stream
workflow was slower than frozen GPU throughput on all eight ATLAS systems.

Approximate RMSD/RMSF are explicitly labelled approximate. Sampled reference
transforms used frozen exact weights and an independent evaluator; evaluation
time is excluded. Synthetic stitched controls are computational stress tests,
not biological continuity evidence. Historical mixed-scale controls are
labelled runtime-only.

No public CLI/default, manuscript, historical result, source dataset or Git
metadata was changed. Raw trajectories and frozen baseline files are excluded.
""",
    )
    write_text(
        bundle / "known_limitations.md",
        """# Known Limitations

- Frozen ATLAS caches did not contain transforms; sampled reference transforms
  were independently reconstructed from frozen exact weights for evaluation.
- Sampled RMSF uses 512 deterministic frames and is not full exact RMSF.
- Internal interval audits detect interpolation error but can miss biased
  landmark sets, as shown by 3nci_A and 5vz0_A.
- Sparse fitting used CPU SVD in formal runs because small batches were faster
  than GPU Horn on the available Quadro P5000.
- Synthetic controls are not continuous biological trajectories; three legacy
  mixed-scale controls are runtime stress evidence only.
- No stream contract is recommended for public use.
""",
    )


def _scan(bundle: Path) -> tuple[list[dict[str, str]], int]:
    issues: list[dict[str, str]] = []
    public_urls = 0
    absolute = re.compile(
        r"(?<![A-Za-z0-9_])/(?:home|storage|tmp|mnt|Users|private|var/tmp)/[^\s`\"']+"
    )
    identity = re.compile(r"\b(?:jjxu|junjie)\b", re.IGNORECASE)
    secrets = (
        re.compile(r"hf_[A-Za-z0-9]{20,}"),
        re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
        re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
        re.compile(r"(?i)(?:password|api[_-]?key|access[_-]?token)\s*[:=]\s*[^\s]{8,}"),
    )
    urls = re.compile(r"https?://[^\s)>\]`\"']+")
    for path in sorted(item for item in bundle.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        relative = str(path.relative_to(bundle))
        for label, pattern in (("absolute_local_path", absolute), ("username", identity)):
            for match in pattern.finditer(text):
                issues.append({"path": relative, "issue": label, "match": match.group(0)[:100]})
        for pattern in secrets:
            if pattern.search(text):
                issues.append({"path": relative, "issue": "secret_or_credential", "match": "redacted"})
        for match in urls.finditer(text):
            parsed = urlparse(match.group(0))
            host = (parsed.hostname or "").lower()
            private = (
                host in {"localhost", "127.0.0.1", "0.0.0.0"}
                or host.endswith(".internal")
                or host.startswith(("10.", "192.168."))
                or "@" in parsed.netloc
                or any(value in parsed.query.lower() for value in ("token=", "key=", "password="))
            )
            if private:
                issues.append({"path": relative, "issue": "private_url", "match": match.group(0)[:100]})
            else:
                public_urls += 1
    return issues, public_urls


def _manifests(bundle: Path) -> None:
    file_manifest = bundle / "file_manifest.tsv"
    hash_manifest = bundle / "sha256_manifest.tsv"
    files = sorted(
        path
        for path in bundle.rglob("*")
        if path.is_file() and path not in {file_manifest, hash_manifest}
    )
    rows = [
        {
            "relative_path": str(path.relative_to(bundle)),
            "section": path.relative_to(bundle).parts[0],
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in files
    ]
    rows.extend(
        [
            {"relative_path": "file_manifest.tsv", "section": "entry", "size_bytes": "self", "sha256": "see_sha256_manifest"},
            {"relative_path": "sha256_manifest.tsv", "section": "entry", "size_bytes": "self", "sha256": "self_excluded"},
        ]
    )
    write_tsv(file_manifest, rows)
    hash_files = sorted(path for path in bundle.rglob("*") if path.is_file() and path != hash_manifest)
    write_tsv(
        hash_manifest,
        [
            {"sha256": sha256(path), "relative_path": str(path.relative_to(bundle))}
            for path in hash_files
        ],
    )


def _portable(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mtime = 0
    return member


def _archive(bundle: Path, target: Path) -> None:
    with tarfile.open(target, "w:gz", format=tarfile.PAX_FORMAT) as handle:
        handle.add(bundle, arcname="handoff_to_chatgpt", filter=_portable)


def _verify(target: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="stream-align-review-") as temporary:
        root = Path(temporary)
        with tarfile.open(target, "r:gz") as handle:
            members = handle.getmembers()
            roots = {Path(member.name).parts[0] for member in members if member.name}
            unsafe = [
                member.name
                for member in members
                if member.name.startswith("/")
                or ".." in Path(member.name).parts
                or member.issym()
                or member.islnk()
            ]
            if roots != {"handoff_to_chatgpt"} or unsafe:
                raise RuntimeError(f"unsafe or non-self-contained archive: {roots}, {unsafe[:3]}")
            if not all(
                member.uid == 0
                and member.gid == 0
                and not member.uname
                and not member.gname
                and member.mtime == 0
                for member in members
            ):
                raise RuntimeError("host tar metadata was not removed")
            handle.extractall(root, filter="data")
        bundle = root / "handoff_to_chatgpt"
        with (bundle / "sha256_manifest.tsv").open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        failures = []
        for row in rows:
            path = bundle / row["relative_path"]
            if not path.is_file() or sha256(path) != row["sha256"]:
                failures.append(row["relative_path"])
        files = {str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()}
        expected = {row["relative_path"] for row in rows} | {"sha256_manifest.tsv"}
        if files != expected:
            failures.extend(sorted(files ^ expected))
        missing = [path for path in _required() if not (bundle / path).is_file()]
        forbidden = [
            str(path.relative_to(bundle))
            for path in bundle.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".h5", ".hdf5", ".xtc", ".dcd", ".tpr", ".npy", ".npz", ".whl"}
        ]
        oversized = [
            str(path.relative_to(bundle))
            for path in bundle.rglob("*")
            if path.is_file() and path.stat().st_size > 100 * 1024 * 1024
        ]
        issues, public_urls = _scan(bundle)
        if failures or missing or forbidden or oversized or issues:
            raise RuntimeError(
                json.dumps(
                    {"hash_failures": failures, "missing": missing, "forbidden": forbidden, "oversized": oversized, "scan": issues},
                    indent=2,
                )
            )
        return {
            "included_files": len(files),
            "hashes_verified": len(rows),
            "required_files": len(_required()),
            "public_urls": public_urls,
            "single_root": "handoff_to_chatgpt",
            "portable_tar_metadata": "pass",
        }


def main() -> None:
    if ARCHIVE.exists() or VALIDATION.exists():
        raise RuntimeError("review archive already exists; refusing to overwrite")
    phase = read_tsv(OUT / "phase_status.tsv")
    for row in phase:
        if row["phase"] == "S8":
            row["status"] = "complete"
            row["evidence"] = ARCHIVE.name
    with tempfile.TemporaryDirectory(prefix="stream-align-stage-") as temporary:
        bundle = Path(temporary) / "handoff_to_chatgpt"
        bundle.mkdir()
        _entry(bundle)
        _stage_outputs(bundle)
        write_tsv(bundle / "phase_status.tsv", phase)
        _stage_source(bundle)
        write_text(
            bundle / "archive_validation_report.md",
            """# Archive Validation

All required sections were present before packaging. Text safety, forbidden
format, 100 MB, single-root, portable metadata, extraction and per-file hash
checks are mandatory builder gates. The adjacent complete-archive SHA-256
protects `sha256_manifest.tsv`, which is self-excluded by definition.
""",
        )
        generated_manifests = {"file_manifest.tsv", "sha256_manifest.tsv"}
        missing = [
            path
            for path in _required()
            if path not in generated_manifests and not (bundle / path).is_file()
        ]
        if missing:
            raise RuntimeError(f"required files missing before packaging: {missing}")
        issues, _ = _scan(bundle)
        if issues:
            raise RuntimeError(f"pre-manifest safety scan failed: {issues[:5]}")
        _manifests(bundle)
        issues, _ = _scan(bundle)
        if issues:
            raise RuntimeError(f"final safety scan failed: {issues[:5]}")
        preflight = Path(temporary) / "preflight.tar.gz"
        _archive(bundle, preflight)
        _verify(preflight)
        _archive(bundle, ARCHIVE)
    verified = _verify(ARCHIVE)
    write_tsv(OUT / "phase_status.tsv", phase)
    archive_hash = sha256(ARCHIVE)
    CHECKSUM.write_text(f"{archive_hash}  {ARCHIVE.name}\n")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_size_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": archive_hash,
        "all_required_sections_present": True,
        "secret_and_absolute_path_scan": "pass",
        "archive_extraction": "pass",
        "manifest_verification": "pass",
        "files_over_100MB": 0,
        "forbidden_raw_or_cache_files": 0,
        "decision": "no stream contract passed; branch stopped",
        "intentional_omissions": [
            "raw ATLAS/mdCATH trajectories and source archives",
            "HDF5/XTC/DCD/TPR/PDB trajectory inputs",
            "NPY/NPZ coordinate and result caches",
            "frozen baseline outputs already registered elsewhere",
            "virtual environments, Python caches, Git internals and secrets",
        ],
        **verified,
    }
    write_json(VALIDATION, result)
    write_text(
        OUT / f"stream_align_v1_review_bundle_{DATE}_validation.md",
        f"""# stream_align_v1 Review Archive Validation

- Archive: `{result['archive']}`
- Size: {result['archive_size_bytes']} bytes
- Included files: {result['included_files']}
- SHA-256: `{archive_hash}`
- Required files: {result['required_files']}/{result['required_files']}
- Secret/absolute-path/private-URL scan: pass
- Extraction and internal manifest verification: pass
- Portable tar metadata: pass
- Files over 100 MB: 0
- Raw trajectories/caches: excluded
""",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
