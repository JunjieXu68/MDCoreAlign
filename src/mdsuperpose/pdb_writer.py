"""Allocation-lean streaming writers for CA-only PDB trajectories."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class StreamingCAPDBWriter:
    """Reuse fixed PDB text fields and one numeric line buffer across frames."""

    def __init__(self, n_residues: int) -> None:
        n_residues = int(n_residues)
        if n_residues < 1:
            raise ValueError("n_residues must be positive")
        if n_residues > 9999:
            raise ValueError(
                "CA-only PDB output supports at most 9999 residues per model"
            )
        lines = [
            (
                f"ATOM  {index:5d}  CA  ALA A{index:4d}    "
                f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}"
                "  1.00  0.00           C\n"
            ).encode("ascii")
            for index in range(1, n_residues + 1)
        ]
        line_widths = {len(line) for line in lines}
        if len(line_widths) != 1:
            raise RuntimeError("internal PDB line width is not fixed")
        self.n_residues = n_residues
        self.line_width = line_widths.pop()
        self._template = np.frombuffer(
            b"".join(lines),
            dtype=np.uint8,
        ).reshape(n_residues, self.line_width)
        self._line_buffer = self._template.copy()

    def _format_frame(self, frame: np.ndarray) -> memoryview:
        """Fill and return the reusable fixed-width atom-line buffer."""

        values = np.asarray(frame, dtype=np.float64)
        if values.shape != (self.n_residues, 3):
            raise ValueError("frame must have shape (n_residues, 3)")
        if not np.isfinite(values).all():
            raise ValueError("frame coordinates must be finite")
        if np.any(np.abs(values) >= 10000.0):
            raise ValueError(
                "PDB 8.3 coordinate fields require absolute values < 10000"
            )
        formatted = np.char.mod("%8.3f", values).astype("S8")
        encoded = formatted.view(np.uint8).reshape(
            self.n_residues,
            3,
            8,
        )
        self._line_buffer[:, 30:38] = encoded[:, 0]
        self._line_buffer[:, 38:46] = encoded[:, 1]
        self._line_buffer[:, 46:54] = encoded[:, 2]
        return memoryview(self._line_buffer)

    def write(self, coords: np.ndarray, path: str | Path) -> None:
        """Stream one multi-model trajectory without per-atom Python objects."""

        values = np.asarray(coords, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (
            self.n_residues,
            3,
        ):
            raise ValueError(
                "coords must have shape (n_frames, n_residues, 3)"
            )
        if values.shape[0] < 1:
            raise ValueError("coords must contain at least one frame")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb", buffering=1024 * 1024) as handle:
            for model_index, frame in enumerate(values, start=1):
                handle.write(f"MODEL     {model_index}\n".encode("ascii"))
                handle.write(self._format_frame(frame))
                handle.write(b"TER\nENDMDL\n")
            handle.write(b"END\n")


def write_multimodel_ca_pdb_streaming(
    coords: np.ndarray,
    path: str | Path,
) -> None:
    """Write a CA-only multi-model PDB using a reusable streaming buffer."""

    values = np.asarray(coords)
    if values.ndim != 3:
        raise ValueError("coords must have shape (n_frames, n_residues, 3)")
    StreamingCAPDBWriter(values.shape[1]).write(values, path)

