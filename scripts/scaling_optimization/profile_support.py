"""Shared lightweight utilities for Phase 0/1 scaling profiling."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import threading
import time
from typing import Iterator

import numpy as np


STAGE_NAMES = (
    "input_open",
    "topology_load",
    "trajectory_decode",
    "preprocess_pbc",
    "atom_selection",
    "reference_selection",
    "candidate_core_initialisation",
    "core_update_iterations",
    "alignment_fit",
    "transform_apply",
    "rmsd_compute",
    "rmsf_compute",
    "flexible_segment_calling",
    "diagnostics",
    "output_serialisation",
    "aligned_trajectory_write",
    "host_to_device",
    "gpu_compute",
    "device_to_host",
    "total",
)


def rss_bytes(pid: int) -> int:
    """Read resident memory for one Linux process."""

    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return 0
    return 0


def child_pids(pid: int) -> list[int]:
    """Return recursively discovered Linux child process IDs."""

    discovered: list[int] = []
    pending = [pid]
    seen = {pid}
    while pending:
        current = pending.pop()
        path = Path(f"/proc/{current}/task/{current}/children")
        try:
            children = [int(value) for value in path.read_text().split()]
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            children = []
        for child in children:
            if child not in seen:
                seen.add(child)
                discovered.append(child)
                pending.append(child)
    return discovered


def process_tree_rss_bytes(pid: int) -> int:
    """Return summed RSS for one process and currently visible children."""

    return sum(rss_bytes(candidate) for candidate in [pid, *child_pids(pid)])


@dataclass(slots=True)
class MemorySampler:
    """Poll process-tree RSS only while an explicit profiling run is active."""

    interval_seconds: float = 0.01
    pid: int = field(default_factory=os.getpid)
    initial_rss_bytes: int = 0
    final_rss_bytes: int = 0
    peak_rss_bytes: int = 0
    sample_count: int = 0
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        """Start polling in a daemon thread."""

        self.initial_rss_bytes = process_tree_rss_bytes(self.pid)
        self.peak_rss_bytes = self.initial_rss_bytes
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll, name="mdca-memory-profiler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop polling and retain final/peak values."""

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 5.0))
        self.final_rss_bytes = process_tree_rss_bytes(self.pid)
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.final_rss_bytes)

    def _poll(self) -> None:
        while not self._stop_event.is_set():
            value = process_tree_rss_bytes(self.pid)
            self.peak_rss_bytes = max(self.peak_rss_bytes, value)
            self.sample_count += 1
            self._stop_event.wait(self.interval_seconds)

    def to_dict(self) -> dict[str, int | float | None]:
        """Return JSON-ready memory measurements."""

        return {
            "initial_rss_bytes": self.initial_rss_bytes,
            "final_rss_bytes": self.final_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mb": self.peak_rss_bytes / (1024**2),
            "rss_delta_bytes": self.final_rss_bytes - self.initial_rss_bytes,
            "sample_count": self.sample_count,
            "peak_gpu_allocated_bytes": None,
            "peak_gpu_reserved_bytes": None,
        }


@dataclass(slots=True)
class StageRecorder:
    """Accumulate high-resolution wall-clock stage timings."""

    enabled: bool = True
    values: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in STAGE_NAMES})

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one named stage when instrumentation is enabled."""

        if name not in self.values:
            self.values[name] = 0.0
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.values[name] += time.perf_counter() - start

    def add(self, name: str, seconds: float | int | None) -> None:
        """Add an already measured duration."""

        if seconds is None:
            return
        self.values[name] = self.values.get(name, 0.0) + max(0.0, float(seconds))

    def to_dict(self) -> dict[str, float]:
        """Return all required stages plus any extra measured stages."""

        return dict(self.values)


def array_record(name: str, array: np.ndarray | None) -> dict[str, object] | None:
    """Describe one material array and whether it owns its storage."""

    if array is None:
        return None
    value = np.asarray(array)
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "nbytes": int(value.nbytes),
        "owns_data": bool(value.flags.owndata),
        "is_view": bool(value.base is not None),
        "base_type": type(value.base).__name__ if value.base is not None else None,
    }


def jsonable(value: object) -> object:
    """Convert common NumPy and Path values to JSON-compatible objects."""

    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write one deterministic JSON profiling record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n")
