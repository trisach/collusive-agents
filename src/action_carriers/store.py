"""Append-only episode storage with resume (one run directory per model).

Layout of ``runs/<mode>/<model-slug>/``:
  manifest.json    Everything that must not change within a run (model, sampling config,
                   prompt version, task config). Resuming with a different manifest fails.
  episodes.jsonl   One JSON line per COMPLETED episode. Written with flush + fsync.
  errors.jsonl     Episodes that failed (e.g. API errors after all retries). Not "done":
                   a resumed run tries them again.
"""

from __future__ import annotations

import json
import os
import pathlib


class EpisodeStore:
    """Run directory with an append-only episode log keyed by (arm, seed)."""

    def __init__(self, run_dir: pathlib.Path, manifest: dict):
        """Opens or creates the run directory.

        Args:
            run_dir: The run directory.
            manifest: Run invariants. Must equal the stored manifest when resuming.

        Raises:
            ValueError: If the directory holds a run with a different manifest.
        """
        self.run_dir = run_dir
        self.episodes_path = run_dir / "episodes.jsonl"
        self.errors_path = run_dir / "errors.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text())
            if stored != manifest:
                changed = sorted(k for k in stored.keys() | manifest.keys()
                                 if stored.get(k) != manifest.get(k))  # fmt: skip
                raise ValueError(
                    f"{run_dir} holds a run with a different manifest (changed: {changed}). "
                    "Use a new --out directory."
                )
        else:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self._drop_partial_last_line()

    def _drop_partial_last_line(self) -> None:
        """A crash during a write can leave a line without its newline; remove it."""
        if not self.episodes_path.exists():
            return
        data = self.episodes_path.read_bytes()
        if data and not data.endswith(b"\n"):
            self.episodes_path.write_bytes(data[: data.rfind(b"\n") + 1])

    def records(self) -> list[dict]:
        """All completed episode records, in completion order."""
        if not self.episodes_path.exists():
            return []
        with self.episodes_path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def done(self) -> set[tuple[str, int]]:
        """(arm, seed) pairs that are complete."""
        return {(r["arm"], r["seed"]) for r in self.records()}

    def append(self, record: dict) -> None:
        """Appends one completed episode durably."""
        _append_line(self.episodes_path, record)

    def log_error(self, info: dict) -> None:
        """Records a failed episode attempt."""
        _append_line(self.errors_path, info)


def _append_line(path: pathlib.Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
