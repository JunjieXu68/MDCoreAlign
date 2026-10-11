"""Shared helpers for the mdCATH benchmark scripts.

The mdCATH dataset is very large, so these helpers default to metadata
inspection, deterministic manifests, and on-demand domain caching instead of
implicit full-dataset download.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
from typing import Any, Iterable

import numpy as np


HF_REPO_ID = "compsciencelab/mdCATH"
HF_REPO_TYPE = "dataset"
TEMPERATURES_K = (320, 348, 379, 413, 450)
REPLICAS = (1, 2, 3, 4, 5)
RESULT_ROOT = Path("results/mdcath_benchmark")
MANIFEST_PATH = RESULT_ROOT / "manifest/mdcath_trajectory_manifest.tsv"
SCHEMA_REPORT_PATH = RESULT_ROOT / "schema_inspection/mdcath_h5_schema_report.md"
LICENSE = "cc-by-4.0"
MDCATH_EXPECTED_DOMAINS = 5398
MDCATH_EXPECTED_TRAJECTORIES = 134950

FAILURE_CATEGORIES = (
    "download_failed",
    "h5_open_failed",
    "schema_unrecognized",
    "missing_coordinates",
    "missing_atom_metadata",
    "ca_selection_failed",
    "too_few_ca_atoms",
    "too_few_frames",
    "coordinate_nan_or_inf",
    "coordinate_unit_suspect",
    "mdcorealign_exception",
    "mdcorealign_timeout",
    "mdcorealign_nonconverged",
    "fullset_baseline_failed",
    "mdlovofit_failed",
    "theseus_failed",
    "output_validation_failed",
    "unknown",
)

MANIFEST_COLUMNS = (
    "domain_id",
    "cath_class",
    "cath_architecture",
    "cath_topology",
    "cath_homologous_superfamily",
    "cath_s35_or_s20_if_available",
    "domain_length",
    "n_atoms",
    "n_residues",
    "temperature_K",
    "replica",
    "trajectory_id",
    "expected_n_frames",
    "coordinate_path",
    "metadata_source",
    "h5_file",
    "local_cache_path",
    "hf_path_or_url",
    "license",
    "mdCATH_version_or_snapshot",
    "include_in_domain_canonical_benchmark",
    "include_in_full_trajectory_benchmark",
    "include_in_external_baseline_subset",
    "notes",
)

PER_TRAJECTORY_COLUMNS = (
    "trajectory_id",
    "domain_id",
    "temperature_K",
    "replica",
    "status",
    "failure_reason",
    "n_frames",
    "n_residues",
    "n_atoms",
    "runtime_seconds",
    "algorithm_runtime_seconds",
    "io_runtime_seconds",
    "peak_memory_mb",
    "iteration_count_mean",
    "converged_fraction",
    "weighted_RMSD",
    "unified_RMSD",
    "fullset_unified_RMSD",
    "delta_unified_RMSD_vs_fullset",
    "robust_RMSF_mean",
    "robust_RMSF_median",
    "ordinary_RMSF_mean",
    "ordinary_RMSF_median",
    "robust_to_ordinary_RMSF_ratio",
    "stable_core_fraction",
    "mobile_fraction",
    "n_core_segments",
    "core_fragmentation_score",
    "noncore_core_RMSF_ratio",
    "RMSF_mobility_spearman",
    "RMSF_mobility_spearman_pvalue",
    "min_weight",
    "max_weight",
    "mean_weight",
    "median_weight",
    "summary_json",
    "profile_tsv",
    "notes",
)


@dataclass(frozen=True)
class H5DependencyStatus:
    """Availability state for optional HDF5 and HF dependencies."""

    h5py_available: bool
    huggingface_hub_available: bool
    h5py_error: str | None
    huggingface_hub_error: str | None


def now_utc() -> str:
    """Return an ISO timestamp in UTC."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_result_dirs() -> None:
    """Create the mdCATH benchmark result directory tree."""

    for relative in (
        "manifest",
        "schema_inspection",
        "logs",
        "per_trajectory",
        "aggregate",
        "figures",
        "failures",
        "external_baselines_subset",
        "manuscript_tables",
        "manuscript_text",
    ):
        (RESULT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def check_optional_dependencies() -> H5DependencyStatus:
    """Check optional dependencies used by mdCATH scripts."""

    errors: dict[str, str | None] = {"h5py": None, "huggingface_hub": None}
    for module_name in errors:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment-specific
            errors[module_name] = str(exc)
    return H5DependencyStatus(
        h5py_available=errors["h5py"] is None,
        huggingface_hub_available=errors["huggingface_hub"] is None,
        h5py_error=errors["h5py"],
        huggingface_hub_error=errors["huggingface_hub"],
    )


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON with stable key order."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n")


def json_ready(value: Any) -> Any:
    """Convert numpy and pathlib objects into JSON-compatible objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_tsv(path: str | Path) -> list[dict[str, str]]:
    """Read a TSV table into dictionaries."""

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: str | Path, rows: Iterable[dict[str, Any]], columns: Iterable[str] | None = None) -> None:
    """Write dictionaries to a TSV table."""

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    columns = list(columns)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        if columns:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key, "")) for key in columns})


def append_tsv(path: str | Path, row: dict[str, Any], columns: Iterable[str]) -> None:
    """Append one row to a TSV table, writing the header if needed."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(columns)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({key: format_cell(row.get(key, "")) for key in columns})


def format_cell(value: Any) -> str:
    """Format a table cell without hiding missing values."""

    if value is None:
        return "NA"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NA"
        return f"{value:.10g}"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_ready(value), sort_keys=True)
    return str(value)


def parse_domain_id_from_h5_path(path: str | Path) -> str:
    """Extract an mdCATH domain ID from a canonical HDF5 file path."""

    name = Path(path).name
    match = re.search(r"mdcath_dataset_([A-Za-z0-9]+)\.h5$", name)
    if match:
        return match.group(1)
    stem = Path(path).stem
    return stem.replace("mdcath_dataset_", "")


def trajectory_id(domain_id: str, temperature_k: int | str, replica: int | str) -> str:
    """Build the deterministic mdCATH trajectory ID."""

    return f"{domain_id}__T{temperature_k}__R{replica}"


def parse_temperature_replica_from_path(path: str) -> tuple[int | None, int | None]:
    """Best-effort extraction of temperature and replica labels from HDF5 paths."""

    lowered = path.lower()
    direct = re.search(r"/(\d{3})/([0-4])/(?:coords|coordinates|positions?)$", lowered)
    if direct:
        return int(direct.group(1)), int(direct.group(2)) + 1
    temperature = None
    replica = None
    for pattern in (r"t(?:emp)?[_-]?(\d{3})", r"temperature[_-]?(\d{3})", r"/(\d{3})k?[/_-]"):
        match = re.search(pattern, lowered)
        if match:
            temperature = int(match.group(1))
            break
    for pattern in (r"r(?:eplica)?[_-]?([1-5])", r"replica[_-]?([1-5])", r"/rep([1-5])"):
        match = re.search(pattern, lowered)
        if match:
            replica = int(match.group(1))
            break
    return temperature, replica


def import_h5py():
    """Import h5py with a clear error message."""

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("h5py is required for mdCATH HDF5 operations. Install with `pip install h5py`.") from exc
    return h5py


def import_huggingface_hub():
    """Import huggingface_hub with a clear error message."""

    try:
        import huggingface_hub  # type: ignore
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "huggingface_hub is required for mdCATH remote listing/download. "
            "Install with `pip install huggingface_hub`."
        ) from exc
    return huggingface_hub


def list_hf_h5_files(repo_id: str = HF_REPO_ID) -> list[str]:
    """List mdCATH HDF5 domain files from Hugging Face without downloading them."""

    hub = import_huggingface_hub()
    files = hub.list_repo_files(repo_id, repo_type=HF_REPO_TYPE)
    return sorted(path for path in files if path.startswith("data/mdcath_dataset_") and path.endswith(".h5"))


def download_hf_file(
    repo_file: str,
    *,
    cache_dir: str | Path | None = None,
    repo_id: str = HF_REPO_ID,
) -> Path:
    """Download one file from Hugging Face into the local cache."""

    hub = import_huggingface_hub()
    local_path = hub.hf_hub_download(
        repo_id=repo_id,
        repo_type=HF_REPO_TYPE,
        filename=repo_file,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(local_path)


def find_local_h5_files(roots: Iterable[str | Path]) -> list[Path]:
    """Find mdCATH HDF5 files under a small set of roots."""

    files: list[Path] = []
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        if root.is_file() and root.name.endswith(".h5"):
            files.append(root)
            continue
        for candidate in root.glob("**/mdcath_dataset_*.h5"):
            files.append(candidate)
    return sorted(set(files))


def default_search_roots() -> list[Path]:
    """Return conservative local search roots for mdCATH data."""

    roots = []
    for env_var in ("MDCATH_ROOT", "MDCATH_CACHE", "HF_HOME"):
        value = os.environ.get(env_var)
        if value:
            roots.append(Path(value))
    roots.extend(
        [
            Path("data"),
            Path("results/mdcath_benchmark/cache"),
            Path("/storage/junjie/projects/DBs"),
            Path("/storage/junjie/projects/Fluctome"),
            Path("/storage/junjie/projects/mdlovofit/data"),
            Path("/storage/junjie/projects/MDCoreAlign/data"),
        ]
    )
    return roots


def iter_h5_datasets(handle: Any) -> Iterable[tuple[str, Any]]:
    """Yield every dataset in an HDF5 file."""

    def visitor(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            datasets.append((f"/{name}", obj))

    datasets: list[tuple[str, Any]] = []
    handle.visititems(visitor)
    return datasets


def dataset_preview(dataset: Any, max_items: int = 5) -> list[Any]:
    """Return a small preview from a dataset without loading large arrays."""

    try:
        if dataset.shape == ():
            return [truncate_preview(decode_value(dataset[()]))]
        if len(dataset.shape) == 0:
            return [truncate_preview(decode_value(dataset[()]))]
        count = min(max_items, int(dataset.shape[0]))
        if count == 0:
            return []
        data = dataset[:count]
        flat = np.asarray(data).reshape(-1)
        return [truncate_preview(decode_value(item)) for item in flat[:max_items]]
    except Exception:
        return []


def truncate_preview(value: Any, max_chars: int = 240) -> Any:
    """Truncate long string previews for human-readable reports."""

    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def decode_value(value: Any) -> Any:
    """Decode bytes and numpy scalar values for reports."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def is_numeric_dataset(dataset: Any) -> bool:
    """Return whether an HDF5 dataset has a numeric dtype."""

    try:
        return np.issubdtype(dataset.dtype, np.number)
    except Exception:
        return False


def find_coordinate_datasets(handle: Any) -> list[dict[str, Any]]:
    """Find likely coordinate arrays from an HDF5 file."""

    candidates: list[dict[str, Any]] = []
    for path, dataset in iter_h5_datasets(handle):
        if not is_numeric_dataset(dataset):
            continue
        shape = tuple(int(dim) for dim in dataset.shape)
        if len(shape) >= 3 and shape[-1] == 3:
            name_score = 1 if re.search(r"(coord|position|xyz|trajectory|pos)", path, re.I) else 0
            candidates.append(
                {
                    "path": path,
                    "shape": shape,
                    "dtype": str(dataset.dtype),
                    "name_score": name_score,
                    "attrs": attrs_to_dict(dataset.attrs),
                }
            )
    candidates.sort(key=lambda item: (-item["name_score"], item["path"]))
    return candidates


def attrs_to_dict(attrs: Any) -> dict[str, Any]:
    """Convert HDF5 attrs to a JSON-ready dictionary."""

    output: dict[str, Any] = {}
    for key in attrs.keys():
        try:
            output[str(key)] = decode_value(attrs[key])
        except Exception:
            output[str(key)] = "<unreadable>"
    return json_ready(output)


def find_string_datasets(handle: Any, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    """Find small string-like datasets matching any path pattern."""

    results: list[dict[str, Any]] = []
    regex = re.compile("|".join(patterns), re.I)
    for path, dataset in iter_h5_datasets(handle):
        if not regex.search(path):
            continue
        preview = dataset_preview(dataset, max_items=8)
        results.append(
            {
                "path": path,
                "shape": tuple(int(dim) for dim in dataset.shape),
                "dtype": str(dataset.dtype),
                "preview": preview,
                "attrs": attrs_to_dict(dataset.attrs),
            }
        )
    return results


def read_dataset_text(dataset: Any, max_chars: int = 5_000_000) -> str | None:
    """Read text from a scalar or array string dataset."""

    try:
        data = dataset[()]
        if isinstance(data, bytes):
            return data[:max_chars].decode("utf-8", errors="replace")
        arr = np.asarray(data)
        if arr.dtype.kind in {"S", "U", "O"}:
            parts = [str(decode_value(item)) for item in arr.reshape(-1)]
            text = "\n".join(parts)
            return text[:max_chars]
    except Exception:
        return None
    return None


def find_pdb_text(handle: Any) -> tuple[str | None, str | None]:
    """Find a PDB-like text block in an HDF5 file."""

    for path, dataset in iter_h5_datasets(handle):
        if not re.search(r"(pdb|topology|structure|protein)", path, re.I):
            continue
        text = read_dataset_text(dataset)
        if text and ("ATOM" in text or "MODEL" in text):
            return path, text
    return None, None


def parse_pdb_ca_metadata(pdb_text: str) -> tuple[np.ndarray, list[str], list[dict[str, str]]]:
    """Parse CA indices and residue metadata from PDB text."""

    ca_indices: list[int] = []
    labels: list[str] = []
    metadata: list[dict[str, str]] = []
    atom_index = 0
    seen_model = False
    for line in pdb_text.splitlines():
        if line.startswith("MODEL"):
            if seen_model:
                break
            seen_model = True
        if line.startswith("ENDMDL") and seen_model:
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        altloc = line[16].strip()
        if atom_name == "CA" and altloc in ("", "A"):
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_id = f"{resseq}{icode}".strip()
            resname = line[17:20].strip() or "UNK"
            ca_indices.append(atom_index)
            labels.append(f"{chain}:{residue_id}:{resname}")
            metadata.append({"chain": chain, "residue_id": residue_id, "resname": resname})
        atom_index += 1
    return np.asarray(ca_indices, dtype=int), labels, metadata


def find_atom_name_ca_indices(handle: Any, n_atoms: int) -> tuple[np.ndarray | None, list[str], list[dict[str, str]], str | None]:
    """Find CA atom indices from atom-name metadata when PDB text is absent."""

    atom_name_paths = find_string_datasets(handle, ("atom.*name", "atom_names", "/name", "atoms"))
    for item in atom_name_paths:
        try:
            dataset = handle[item["path"]]
            data = np.asarray(dataset[()]).reshape(-1)
        except Exception:
            continue
        if data.shape[0] != n_atoms:
            continue
        names = np.asarray([str(decode_value(x)).strip() for x in data], dtype=object)
        ca_mask = names == "CA"
        if np.any(ca_mask):
            indices = np.flatnonzero(ca_mask)
            labels = [f"_:{idx + 1}:UNK" for idx in range(indices.shape[0])]
            metadata = [
                {"chain": "_", "residue_id": str(idx + 1), "resname": "UNK"}
                for idx in range(indices.shape[0])
            ]
            return indices, labels, metadata, item["path"]
    return None, [], [], None


def normalize_coordinate_array(coords: np.ndarray) -> np.ndarray:
    """Normalize a coordinate array to shape (frames, atoms, 3)."""

    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 3 and coords.shape[-1] == 3:
        return coords
    if coords.ndim == 4 and coords.shape[-1] == 3:
        leading = int(np.prod(coords.shape[:-2]))
        return coords.reshape((leading, coords.shape[-2], 3))
    raise ValueError(f"coordinate array must have shape (frames, atoms, 3); got {coords.shape}")


def maybe_convert_units_to_angstrom(coords: np.ndarray, attrs: dict[str, Any] | None = None) -> tuple[np.ndarray, str]:
    """Convert explicit nanometer inputs and preserve explicit Angstrom inputs.

    Coordinate magnitudes are translation-dependent and therefore must not be
    used to guess molecular length units.
    """

    attrs = attrs or {}
    unit_text = " ".join(str(value).lower() for value in attrs.values())
    if "nanometer" in unit_text or re.search(r"\bnm\b", unit_text):
        return coords * 10.0, "converted_nm_to_angstrom"
    if (
        "angstrom" in unit_text
        or "ångström" in unit_text
        or re.search(r"(?:^|\s)[aå](?:\s|$)", unit_text)
    ):
        return coords, "verified_angstrom_metadata"
    return coords, "assumed_angstrom_no_unit_metadata"


def memory_mb() -> float | None:
    """Return current maximum resident set size in MB when available."""

    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(value / (1024 * 1024))
        return float(value / 1024)
    except Exception:
        return None


def write_traceback(path: str | Path, exc: BaseException) -> None:
    """Write an exception traceback to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def cache_size_gb(path: str | Path) -> float:
    """Compute approximate size of a cache directory in GB."""

    path = Path(path)
    if not path.exists():
        return 0.0
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total / (1024**3)


def purge_path(path: str | Path) -> None:
    """Remove a cached file or directory if it exists."""

    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def stable_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """Compute Spearman correlation and p-value when scipy is available."""

    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y, nan_policy="omit")
        corr = float(result.statistic)
        pvalue = float(result.pvalue)
        if not math.isfinite(corr):
            return None, None
        return corr, pvalue if math.isfinite(pvalue) else None
    except Exception:
        return _spearman_no_pvalue(x, y), None


def _spearman_no_pvalue(x: np.ndarray, y: np.ndarray) -> float | None:
    """Small Spearman fallback without p-values."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.size < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    xr -= np.mean(xr)
    yr -= np.mean(yr)
    denom = float(np.sqrt(np.sum(xr * xr) * np.sum(yr * yr)))
    if denom == 0.0:
        return None
    return float(np.sum(xr * yr) / denom)
