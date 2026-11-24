"""Readers for the PhysioNet/CinC Challenge 2012 files.

Each ICU stay is one text file with three columns: Time (HH:MM since ICU
admission), Parameter, Value. The outcomes files hold one row per RecordID.
The readers work whether a set is zipped (set-a.zip), a gzipped tarball
(set-c.tar.gz) or extracted (set-a/ folder), so nothing has to be unpacked
by hand. PhysioNet offers set-a and set-b as .zip but set-c only as .tar.gz.
"""
from __future__ import annotations

import io
import tarfile
import zipfile
from functools import lru_cache
from pathlib import Path

import pandas as pd

SETS = ("a", "b", "c")


def find_source(raw_dir: Path, s: str) -> Path:
    """Return the first existing source for set s: .zip, .tar.gz, then folder."""
    candidates = [
        raw_dir / f"set-{s}.zip",
        raw_dir / f"set-{s}.tar.gz",
        raw_dir / f"set-{s}",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"None of {[str(c) for c in candidates]} exists"
    )


def _kind(src: Path) -> str:
    if src.is_dir():
        return "dir"
    if src.name.endswith(".tar.gz"):
        return "tar"
    return "zip"


@lru_cache(maxsize=None)
def _tar_members(path_str: str) -> dict[str, str]:
    """Map record basename -> member name for the .txt files in a tarball."""
    with tarfile.open(path_str, "r:gz") as tf:
        return {
            Path(m.name).name: m.name
            for m in tf.getmembers()
            if m.isfile()
            and m.name.endswith(".txt")
            and not Path(m.name).name.startswith(".")
        }


def list_records(raw_dir: Path, s: str) -> list[str]:
    """Return the record file names (e.g. '132539.txt') in one set."""
    src = find_source(raw_dir, s)
    kind = _kind(src)
    if kind == "zip":
        with zipfile.ZipFile(src) as zf:
            return sorted(
                Path(n).name for n in zf.namelist()
                if n.endswith(".txt") and not Path(n).name.startswith(".")
            )
    if kind == "tar":
        return sorted(_tar_members(str(src)))
    return sorted(p.name for p in src.rglob("*.txt"))


def read_record(raw_dir: Path, s: str, fname: str) -> pd.DataFrame:
    """Read one ICU stay as a long table: Time, Parameter, Value."""
    src = find_source(raw_dir, s)
    kind = _kind(src)
    if kind == "zip":
        with zipfile.ZipFile(src) as zf:
            member = next(n for n in zf.namelist() if Path(n).name == fname)
            text = zf.read(member).decode("utf-8")
        return pd.read_csv(io.StringIO(text))
    if kind == "tar":
        member = _tar_members(str(src))[fname]
        with tarfile.open(src, "r:gz") as tf:
            text = tf.extractfile(member).read().decode("utf-8")
        return pd.read_csv(io.StringIO(text))
    match = next(src.rglob(fname))
    return pd.read_csv(match)


def read_outcomes(raw_dir: Path, s: str) -> pd.DataFrame:
    """Read Outcomes-<s>.txt and tag each row with its set."""
    df = pd.read_csv(raw_dir / f"Outcomes-{s}.txt")
    df["set"] = s
    return df
