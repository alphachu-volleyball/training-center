"""Experiment metadata collection for reproducibility tracking."""

from __future__ import annotations

import importlib.metadata
import subprocess


def get_experiment_metadata() -> dict[str, str | bool]:
    """Collect metadata to log at the start of every training run.

    Returns dict with:
        commit: git HEAD hash (or "unknown")
        dirty: whether uncommitted changes exist
        pika_zoo_version: installed pika-zoo version
    """
    return {
        "commit": _git_commit(),
        "dirty": _git_dirty(),
        "pika_zoo_version": _pika_zoo_version(),
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(["git", "diff", "--quiet"], capture_output=True)
        return result.returncode != 0
    except FileNotFoundError:
        return False


def _pika_zoo_version() -> str:
    try:
        return importlib.metadata.version("pika-zoo")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
