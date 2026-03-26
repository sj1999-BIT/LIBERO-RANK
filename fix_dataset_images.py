"""
fix_dataset_images.py
=====================
Walks a LIBERO-RANK dataset directory, finds every .hdr5 file, and corrects
the image arrays that were stored after prep_for_display() was incorrectly
applied during collection (leaving frames BGR + vertically flipped instead of
RGB + vertically flipped as LIBERO convention requires).

Fixes applied to every episode in every file:
    obs/agentview_rgb      (T, H, W, 3) uint8   BGR+flipped  →  RGB+flipped
    obs/eye_in_hand_rgb    (T, H, W, 3) uint8   same
    obs/agentview_depth    (T, H, W, 1) float32 flipped      →  un-flipped
    obs/eye_in_hand_depth  (T, H, W, 1) float32 same

All other datasets and attributes are left untouched.

The fix is done in-place: each dataset is read, corrected, and written back
with the same shape/dtype. No temporary files are created.

Usage
-----
    python fix_dataset_images.py /Hyperplane/shuijie/hdf5_trajectory_data

    # dry run — print what would be changed without writing
    python fix_dataset_images.py /Hyperplane/shuijie/hdf5_trajectory_data --dry-run

    # single file
    python fix_dataset_images.py /path/to/demo_0.hdr5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Per-array transforms
# ─────────────────────────────────────────────────────────────────────────────

def fix_rgb(arr: np.ndarray) -> np.ndarray:
    """
    (T, H, W, 3) uint8, BGR + vertically flipped
    →  (T, H, W, 3) uint8, RGB + vertically flipped  (LIBERO convention)

    prep_for_display did:  channel swap (RGB→BGR) then flipud
    We undo in the same order: flipud then channel swap (BGR→RGB)
    Both ops are self-inverse so applying them once more undoes both.
    """
    return np.ascontiguousarray(arr[:, ::-1, :, ::-1])


def fix_depth(arr: np.ndarray) -> np.ndarray:
    """
    (T, H, W, 1) or (T, H, W) float32, vertically flipped
    →  un-flipped

    prep_for_display did flipud on depth too (no channel swap for single-channel).
    """
    return np.ascontiguousarray(arr[:, ::-1])


# ─────────────────────────────────────────────────────────────────────────────
# Per-file fixer
# ─────────────────────────────────────────────────────────────────────────────

RGB_KEYS   = ["obs/agentview_rgb",     "obs/eye_in_hand_rgb"]
DEPTH_KEYS = ["obs/agentview_depth",   "obs/eye_in_hand_depth"]


def fix_episode(ep_grp: h5py.Group, dry_run: bool) -> bool:
    """
    Fix all image datasets inside one episode group.
    Returns True if any dataset was (or would be) changed.
    """
    changed = False

    for key in RGB_KEYS:
        if key not in ep_grp:
            print(f"    [SKIP] {key} not found")
            continue
        arr = ep_grp[key][()]
        fixed = fix_rgb(arr)
        if not dry_run:
            ep_grp[key][...] = fixed
        changed = True
        print(f"    {'[DRY]' if dry_run else '[FIX]'} {key}  {arr.shape}  {arr.dtype}")

    for key in DEPTH_KEYS:
        if key not in ep_grp:
            print(f"    [SKIP] {key} not found")
            continue
        arr = ep_grp[key][()]
        fixed = fix_depth(arr)
        if not dry_run:
            ep_grp[key][...] = fixed
        changed = True
        print(f"    {'[DRY]' if dry_run else '[FIX]'} {key}  {arr.shape}  {arr.dtype}")

    return changed


def fix_file(path: Path, dry_run: bool) -> dict:
    """
    Open one .hdr5 file and fix all episodes inside it.
    Returns a summary dict.
    """
    mode = "r" if dry_run else "r+"
    summary = {"path": path, "episodes": 0, "fixed": 0, "errors": 0}

    try:
        with h5py.File(str(path), mode) as f:
            if "data" not in f:
                print(f"  [WARN] no 'data' group in {path}")
                return summary

            episodes = sorted(f["data"].keys())
            summary["episodes"] = len(episodes)

            for ep_name in episodes:
                print(f"  {ep_name}")
                try:
                    changed = fix_episode(f[f"data/{ep_name}"], dry_run)
                    if changed:
                        summary["fixed"] += 1
                except Exception as e:
                    print(f"    [ERROR] {e}")
                    summary["errors"] += 1

    except Exception as e:
        print(f"  [ERROR] could not open {path}: {e}")
        summary["errors"] += 1

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_hdr5_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix == ".hdr5":
        return [root]
    return sorted(root.rglob("*.hdr5"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fix BGR+flip bug in LIBERO-RANK .hdr5 files")
    parser.add_argument("path", type=Path,
                        help="dataset root directory OR a single .hdr5 file")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would change without writing anything")
    args = parser.parse_args()

    files = find_hdr5_files(args.path)
    if not files:
        print(f"No .hdr5 files found under {args.path}")
        sys.exit(1)

    print(f"Found {len(files)} file(s)  |  dry_run={args.dry_run}\n")

    total_files    = 0
    total_episodes = 0
    total_fixed    = 0
    total_errors   = 0

    for path in files:
        print(f"{'─'*60}")
        print(f"{path}")
        s = fix_file(path, args.dry_run)
        total_files    += 1
        total_episodes += s["episodes"]
        total_fixed    += s["fixed"]
        total_errors   += s["errors"]

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Files processed : {total_files}")
    print(f"  Episodes seen   : {total_episodes}")
    print(f"  Episodes fixed  : {total_fixed}")
    print(f"  Errors          : {total_errors}")
    if args.dry_run:
        print("\n  [DRY RUN] nothing was written.")


if __name__ == "__main__":
    main()