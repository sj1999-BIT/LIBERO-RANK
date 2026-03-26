"""
visualize_demo.py — LIBERO-RANK Demo Visualizer
================================================
Renders RGB and depth videos from LIBERO-format .hdr5 files.

Two rendering modes
-------------------
  stored   Read frames directly from the HDF5 obs arrays.
           Fast. No LIBERO installation required.

  replay   Reconstruct the environment and re-execute stored actions.
           Requires LIBERO. Supports arbitrary output resolution.
           Also runs a physics divergence check at every step.

Output options
--------------
  --save          Write .mp4 files to disk  (default: on)
  --no-save       Skip writing to disk
  --display       Show frames in a live OpenCV window  (default: off)
  --resolution    Output resolution for replay mode: 128, 256, 512  (default: 256)

Examples
--------
  # stored frames, save videos, no live display
  python visualize_demo.py --hdf5 demo_0.hdr5

  # stored frames, display live, don't save
  python visualize_demo.py --hdf5 demo_0.hdr5 --display --no-save

  # replay mode at 512×512, save + display
  python visualize_demo.py --hdf5 demo_0.hdr5 --mode replay --resolution 512 --display

  # stored, only demo_0, custom output dir
  python visualize_demo.py --hdf5 demo_0.hdr5 --demo demo_0 --out /tmp/vis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import h5py
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Frame helpers  (no LIBERO dependency)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_depth(depth_frame: np.ndarray) -> np.ndarray:
    """
    (H, W, 1) or (H, W) float32  →  uint8 BGR, TURBO colormap.
    Near = blue, Far = red.
    """
    d = depth_frame.squeeze()
    d_min, d_max = d.min(), d.max()
    if d_max - d_min < 1e-6:
        gray = np.zeros_like(d, dtype=np.uint8)
    else:
        gray = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def prep_rgb(frame: np.ndarray, label: str = "") -> np.ndarray:
    """(H, W, 3) uint8 RGB  →  BGR, flipped vertically, optional label."""
    bgr = np.flipud(frame[..., ::-1]).copy()
    if label:
        cv2.putText(bgr, label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


def prep_depth(frame: np.ndarray, label: str = "") -> np.ndarray:
    """(H, W, 1) float32  →  colorized BGR, flipped vertically, optional label."""
    colored = np.flipud(normalize_depth(frame)).copy()
    if label:
        cv2.putText(colored, label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return colored


def make_side_by_side(left: np.ndarray, right: np.ndarray, divider_px: int = 4) -> np.ndarray:
    """Concatenate two same-height BGR frames with a thin black divider."""
    divider = np.zeros((left.shape[0], divider_px, 3), dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def add_instruction_banner(frame: np.ndarray, text: str) -> np.ndarray:
    """Prepend a dark banner with instruction text above the frame."""
    if not text:
        return frame
    banner = np.zeros((24, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, text[:90], (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([banner, frame])


# ─────────────────────────────────────────────────────────────────────────────
# Video I/O  (no LIBERO dependency)
# ─────────────────────────────────────────────────────────────────────────────

class FrameSink:
    """
    Collects frames and optionally writes a video file and/or displays live.

    Args:
        out_path    : destination .mp4 path; ignored when save=False
        fps         : playback frame rate
        save        : write frames to disk
        display     : show frames in an OpenCV window
        window_name : title of the live display window
    """

    def __init__(
        self,
        out_path:    str,
        fps:         int  = 10,
        save:        bool = True,
        display:     bool = False,
        window_name: str  = "LIBERO-RANK",
    ):
        self.out_path    = out_path
        self.fps         = fps
        self.save        = save
        self.display     = display
        self.window_name = window_name

        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_count = 0

    def write(self, frame: np.ndarray) -> None:
        """Accept one BGR frame."""
        if self.save:
            if self._writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    self.out_path, fourcc, self.fps, (w, h))
            self._writer.write(frame)

        if self.display:
            cv2.imshow(self.window_name, frame)
            delay = max(1, int(1000 / self.fps))
            key = cv2.waitKey(delay)
            if key == ord("q"):
                raise KeyboardInterrupt("User quit display.")

        self._frame_count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            size_mb = os.path.getsize(self.out_path) / 1e6
            print(f"  → saved {self._frame_count} frames "
                  f"({size_mb:.1f} MB)  {self.out_path}")

        if self.display:
            cv2.destroyWindow(self.window_name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Mode A — stored frame rendering  (no LIBERO dependency)
# ─────────────────────────────────────────────────────────────────────────────

def render_stored(
    ep_grp:      h5py.Group,
    out_rgb:     str,
    out_depth:   str,
    fps:         int  = 10,
    save:        bool = True,
    display:     bool = False,
    instruction: str  = "",
) -> None:
    """
    Read RGB and depth frames directly from the HDF5 obs arrays and render
    two side-by-side videos (agentview | eye_in_hand).

    No LIBERO installation required.
    """
    agentview_rgb   = ep_grp["obs/agentview_rgb"][()]       # (T, H, W, 3)
    eye_in_hand_rgb = ep_grp["obs/eye_in_hand_rgb"][()]     # (T, H, W, 3)
    T = len(agentview_rgb)

    has_depth = ("obs/agentview_depth" in ep_grp and
                 "obs/eye_in_hand_depth" in ep_grp)

    agentview_depth   = ep_grp["obs/agentview_depth"][()]   if has_depth else None
    eye_in_hand_depth = ep_grp["obs/eye_in_hand_depth"][()] if has_depth else None

    with FrameSink(out_rgb, fps, save, display, "RGB (stored)") as rgb_sink, \
         FrameSink(out_depth, fps, save and has_depth, display and has_depth,
                   "Depth (stored)") as depth_sink:

        for t in range(T):
            # ── RGB ───────────────────────────────────────────────────────
            left  = prep_rgb(agentview_rgb[t],   label=f"agentview  t={t}")
            right = prep_rgb(eye_in_hand_rgb[t], label="eye_in_hand")
            row   = add_instruction_banner(make_side_by_side(left, right), instruction)
            rgb_sink.write(row)

            # ── depth ─────────────────────────────────────────────────────
            if has_depth:
                left  = prep_depth(agentview_depth[t],   label=f"agentview depth  t={t}")
                right = prep_depth(eye_in_hand_depth[t], label="eye_in_hand depth")
                row   = add_instruction_banner(make_side_by_side(left, right), instruction)
                depth_sink.write(row)

    if not has_depth:
        print("  [SKIP depth] no depth arrays in this episode")


# ─────────────────────────────────────────────────────────────────────────────
# Mode B — environment reconstruction  (requires LIBERO)
# ─────────────────────────────────────────────────────────────────────────────

def _build_env(ep_grp: h5py.Group, camera_height: int, camera_width: int):
    """
    Reconstruct a LIBERO OffScreenRenderEnv from the episode's stored
    bddl_content, model_file, and init_state.

    LIBERO import is deferred to this function — the rest of the module
    works without LIBERO installed.

    Returns (env, tmp_bddl_path).  Caller must call env.close() and
    os.unlink(tmp_bddl_path) when done.
    """
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as e:
        raise ImportError(
            "LIBERO is not installed. "
            "Install it or use --mode stored instead."
        ) from e

    bddl_content = ep_grp.attrs.get("bddl_content", None)
    model_xml    = ep_grp.attrs.get("model_file",   None)

    if bddl_content is None:
        raise ValueError(f"No bddl_content attr in {ep_grp.name}")
    if model_xml is None:
        raise ValueError(f"No model_file attr in {ep_grp.name}")

    fd, tmp_bddl_path = tempfile.mkstemp(suffix=".bddl", prefix="libero_replay_")
    os.close(fd)
    with open(tmp_bddl_path, "w") as f:
        f.write(bddl_content)

    env = OffScreenRenderEnv(
        bddl_file_name = tmp_bddl_path,
        camera_heights = camera_height,
        camera_widths  = camera_width,
        camera_depths  = True,
        camera_names   = ["robot0_eye_in_hand", "agentview"],
    )

    env.reset()
    env.reset_from_xml_string(model_xml)

    # populate controller buffers, then restore exact init state
    for robot in env.robots:
        robot._load_controller()
        robot.reset(deterministic=True)

    env.sim.set_state_from_flattened(ep_grp.attrs["init_state"])
    env.sim.forward()

    # sync controller goal to actual post-restore EEF pose
    for robot in env.robots:
        site_id  = env.sim.model.site_name2id(robot.controller.eef_name)
        robot.controller.goal_pos = env.sim.data.site_xpos[site_id].copy()
        robot.controller.goal_ori = (
            env.sim.data.site_xmat[site_id]
            .reshape(3, 3).copy().astype(np.float64)
        )
        if hasattr(robot, "gripper") and hasattr(robot.gripper, "current_action"):
            gripper_qpos = env.sim.data.qpos[
                robot._ref_gripper_joint_pos_indexes].copy()
            robot.gripper.current_action = gripper_qpos[0]

    return env, tmp_bddl_path


def render_replay(
    ep_grp:               h5py.Group,
    out_rgb:              str,
    out_depth:            str,
    fps:                  int   = 10,
    save:                 bool  = True,
    display:              bool  = False,
    instruction:          str   = "",
    camera_height:        int   = 256,
    camera_width:         int   = 256,
    divergence_threshold: float = 0.01,
) -> None:
    """
    Reconstruct the LIBERO environment, replay stored actions step by step,
    and render two side-by-side videos (agentview | eye_in_hand).

    Also logs physics divergence: at each step the replayed sim state is
    compared against the stored state; steps exceeding divergence_threshold
    are printed.

    Requires LIBERO.
    """
    actions = ep_grp["actions"][()]   # (T, 7)
    states  = ep_grp["states"][()]    # (T, state_dim)
    T       = len(actions)

    print(f"  [REPLAY] T={T}  resolution={camera_height}×{camera_width}")

    env, tmp_bddl_path = _build_env(ep_grp, camera_height, camera_width)
    max_divergence = 0.0

    try:
        with FrameSink(out_rgb,   fps, save, display, "RGB (replay)") as rgb_sink, \
             FrameSink(out_depth, fps, save, display, "Depth (replay)") as depth_sink:

            for t, action in enumerate(actions):
                obs, _, _, _ = env.step(action)

                # divergence check
                if t + 1 < len(states):
                    err = np.linalg.norm(
                        states[t + 1] - env.sim.get_state().flatten())
                    max_divergence = max(max_divergence, err)
                    if err > divergence_threshold:
                        print(f"    [DIVERGE] step {t:3d}  err={err:.4f}")

                # RGB
                left  = prep_rgb(obs["agentview_image"],
                                 label=f"agentview (replay) t={t}")
                right = prep_rgb(obs["robot0_eye_in_hand_image"],
                                 label="eye_in_hand (replay)")
                rgb_sink.write(
                    add_instruction_banner(
                        make_side_by_side(left, right), instruction))

                # depth
                left  = prep_depth(obs["agentview_depth"],
                                   label=f"agentview depth (replay) t={t}")
                right = prep_depth(obs["robot0_eye_in_hand_depth"],
                                   label="eye_in_hand depth (replay)")
                depth_sink.write(
                    add_instruction_banner(
                        make_side_by_side(left, right), instruction))

    finally:
        env.close()
        os.unlink(tmp_bddl_path)

    print(f"  [REPLAY] max physics divergence: {max_divergence:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level entry points
# ─────────────────────────────────────────────────────────────────────────────

def visualize_hdf5(
    hdf5_path:     str,
    mode:          str  = "stored",        # "stored" | "replay"
    demo_filter:   Optional[str]  = None,
    out_dir:       Optional[str]  = None,
    fps:           int  = 10,
    save:          bool = True,
    display:       bool = False,
    resolution:    int  = 256,             # used only for mode="replay"
) -> None:
    """
    Main visualization entry point.

    Args:
        hdf5_path   : path to .hdr5 file
        mode        : "stored" reads frames from HDF5; "replay" re-executes
                      actions in the simulator (requires LIBERO)
        demo_filter : render only this episode key, e.g. "demo_0"
        out_dir     : output directory (default: same folder as hdf5_path)
        fps         : video frame rate
        save        : write video files to disk
        display     : show live OpenCV window
        resolution  : output pixel size for replay mode (128 / 256 / 512)
    """
    hdf5_path = str(hdf5_path)
    if out_dir is None:
        out_dir = str(Path(hdf5_path).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(hdf5_path).stem

    with h5py.File(hdf5_path, "r") as f:
        # read shared instruction from file-level attrs
        try:
            problem_info = json.loads(f["data"].attrs["problem_info"])
            file_instruction = problem_info.get("language_instruction", "")
        except Exception:
            file_instruction = ""

        demos = sorted(f["data"].keys())
        if demo_filter is not None:
            if demo_filter not in demos:
                raise ValueError(
                    f"'{demo_filter}' not found. Available: {demos}")
            demos = [demo_filter]

        tag = mode  # "stored" or "replay"
        print(f"\n[{tag.upper()}]  {hdf5_path}")
        print(f"  instruction : {file_instruction}")
        print(f"  demos       : {demos}")
        print(f"  save        : {save}  |  display : {display}")
        if mode == "replay":
            print(f"  resolution  : {resolution}×{resolution}")
        print(f"  output dir  : {out_dir}")

        for ep_name in demos:
            ep_grp      = f[f"data/{ep_name}"]
            T           = ep_grp.attrs.get("num_samples", "?")
            instruction = ep_grp.attrs.get("instruction", file_instruction)

            print(f"\n── {ep_name}  (T={T}) ──────────────")

            out_rgb   = os.path.join(out_dir, f"{stem}_{ep_name}_{tag}_rgb.mp4")
            out_depth = os.path.join(out_dir, f"{stem}_{ep_name}_{tag}_depth.mp4")

            if mode == "stored":
                render_stored(
                    ep_grp      = ep_grp,
                    out_rgb     = out_rgb,
                    out_depth   = out_depth,
                    fps         = fps,
                    save        = save,
                    display     = display,
                    instruction = instruction,
                )
            elif mode == "replay":
                render_replay(
                    ep_grp        = ep_grp,
                    out_rgb       = out_rgb,
                    out_depth     = out_depth,
                    fps           = fps,
                    save          = save,
                    display       = display,
                    instruction   = instruction,
                    camera_height = resolution,
                    camera_width  = resolution,
                )
            else:
                raise ValueError(f"Unknown mode '{mode}'. Use 'stored' or 'replay'.")

    print(f"\n[DONE]  output dir: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# BDDL sanity check
# ─────────────────────────────────────────────────────────────────────────────

def check_bddl_uniqueness(hdf5_path: str) -> None:
    """Print per-demo BDDL fingerprints and report whether all are unique."""
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys())
        print(f"Total demos: {len(demos)}\n")

        bddl_hashes: set = set()
        for ep in demos:
            ep_grp = f[f"data/{ep}"]
            bddl   = ep_grp.attrs.get("bddl_content", "MISSING")
            instr  = ep_grp.attrs.get("instruction",  "MISSING")

            if bddl != "MISSING":
                goal_match = re.search(
                    r":goal\s+\(And\s+\(On\s+(\S+)\s+(\S+)\)\)", bddl)
                obj_inst  = goal_match.group(1) if goal_match else "?"
                bowl_inst = goal_match.group(2) if goal_match else "?"
                obj_type  = obj_inst.rsplit("_",  1)[0]
                bowl_type = bowl_inst.rsplit("_", 1)[0]
                bddl_hash = hashlib.md5(bddl.encode()).hexdigest()[:8]
                bddl_hashes.add(bddl_hash)
            else:
                obj_type = bowl_type = bddl_hash = "?"

            print(f"  {ep}: obj={obj_type:30s} bowl={bowl_type:20s} "
                  f"hash={bddl_hash} | instr='{instr}'")

        n = len(bddl_hashes)
        print(f"\nUnique BDDLs: {n} / {len(demos)}")
        if n == len(demos):
            print("✅ All demos have different BDDLs")
        else:
            print("⚠️  Some demos share the same BDDL")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LIBERO-RANK demo visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--hdf5", type=str, required=True,
                   help="path to .hdr5 file")
    p.add_argument("--demo", type=str, default=None,
                   help="episode key to render (e.g. demo_0). Default: all")
    p.add_argument("--out",  type=str, default=None,
                   help="output directory. Default: same folder as --hdf5")
    p.add_argument("--fps",  type=int, default=10,
                   help="video frame rate (default: 10)")

    # rendering mode
    p.add_argument("--mode", choices=["stored", "replay"], default="stored",
                   help=(
                       "stored : read frames from HDF5 obs arrays (fast, no LIBERO needed). "
                       "replay : reconstruct env and re-execute actions (requires LIBERO)."
                   ))

    # output options
    save_grp = p.add_mutually_exclusive_group()
    save_grp.add_argument("--save",    dest="save", action="store_true",  default=True,
                          help="write video files to disk (default)")
    save_grp.add_argument("--no-save", dest="save", action="store_false",
                          help="skip writing video files")

    p.add_argument("--display", action="store_true", default=False,
                   help="show frames in a live OpenCV window (press q to quit)")

    # replay resolution
    p.add_argument("--resolution", type=int, choices=[128, 256, 512], default=256,
                   help="output resolution for replay mode (default: 256)")

    # utility sub-command
    p.add_argument("--check-bddl", action="store_true",
                   help="print BDDL fingerprints and uniqueness check, then exit")

    return p


def postprocess_demo_images(demo: dict) -> dict:
    """
    Fix image orientation and colour space in a collected demo dict.

    LIBERO obs images are stored upside-down and in RGB.
    prep_for_display() was incorrectly applied during collection, leaving
    frames BGR + flipped (double-wrong). This function corrects both arrays
    to the proper LIBERO storage convention: RGB, upside-down.

    Operates on:
        agentview_rgb      (T, H, W, 3)  uint8  BGR+flipped → RGB+flipped (LIBERO convention)
        eye_in_hand_rgb    (T, H, W, 3)  uint8  same
        agentview_depth    (T, H, W, 1)  float32  no colour transform needed, just un-flip
        eye_in_hand_depth  (T, H, W, 1)  float32  same

    All other keys are passed through unchanged.
    Returns a new dict (does not mutate the input).
    """
    fixed = dict(demo)   # shallow copy — non-image arrays are shared, not duplicated

    def fix_rgb(arr):
        # arr is (T, H, W, 3), currently BGR + flipped
        # flip back vertically, then swap BGR→RGB
        return np.ascontiguousarray(arr[:, ::-1, :, ::-1])

    def fix_depth(arr):
        # arr is (T, H, W, 1) or (T, H, W), currently flipped
        # just un-flip
        return np.ascontiguousarray(arr[:, ::-1])

    fixed["agentview_rgb"]     = fix_rgb(demo["agentview_rgb"])
    fixed["eye_in_hand_rgb"]   = fix_rgb(demo["eye_in_hand_rgb"])
    fixed["agentview_depth"]   = fix_depth(demo["agentview_depth"])
    fixed["eye_in_hand_depth"] = fix_depth(demo["eye_in_hand_depth"])

    return fixed

if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.check_bddl:
        check_bddl_uniqueness(args.hdf5)
    else:
        visualize_hdf5(
            hdf5_path   = args.hdf5,
            mode        = args.mode,
            demo_filter = args.demo,
            out_dir     = args.out,
            fps         = args.fps,
            save        = args.save,
            display     = args.display,
            resolution  = args.resolution,
        )