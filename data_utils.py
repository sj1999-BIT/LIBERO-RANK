"""
HDF5 Demo Visualizer
======================
Given a LIBERO-format HDF5 file, renders videos of:
  1. RGB (agentview + eye_in_hand side by side)
  2. Depth map (agentview + eye_in_hand side by side, colorized)

Usage:
    # visualize all demos in a file
    python visualize_demo.py --hdf5 /path/to/demo.hdf5

    # visualize only demo_0
    python visualize_demo.py --hdf5 /path/to/demo.hdf5 --demo demo_0

    # custom output directory
    python visualize_demo.py --hdf5 /path/to/demo.hdf5 --out /path/to/output

    # custom fps
    python visualize_demo.py --hdf5 /path/to/demo.hdf5 --fps 10
"""

import argparse
import os
import h5py
import numpy as np
import cv2
import tempfile

from pathlib import Path
from libero.libero.envs import TASK_MAPPING
import robosuite.utils.transform_utils as T_utils


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_depth(depth_frame):
    """
    Converts a (H, W, 1) or (H, W) float32 depth frame to a uint8 colorized
    BGR image using the TURBO colormap (vivid, perceptually uniform).

    Near = blue, Far = red.
    """
    d = depth_frame.squeeze()            # (H, W)
    d_min, d_max = d.min(), d.max()

    if d_max - d_min < 1e-6:
        gray = np.zeros_like(d, dtype=np.uint8)
    else:
        gray = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)

    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)   # (H, W, 3) BGR
    return colored


def prep_rgb(frame, label=None):
    """
    Converts a (H, W, 3) uint8 RGB frame to BGR for OpenCV, flips vertically
    (LIBERO stores images upside down), and optionally adds a label.
    """
    # bgr = frame[..., ::-1].copy()       # RGB → BGR
    # bgr = np.flipud(bgr).copy()         # flip vertical
    # if label:
    #     cv2.putText(bgr, label, (5, 18),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    # return bgr

    if label:
        cv2.putText(frame, label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def prep_depth(frame, label=None):
    """
    Converts a (H, W, 1) float32 depth frame to a colorized BGR image,
    flips vertically, and optionally adds a label.
    """
    colored = normalize_depth(frame)
    # colored = np.flipud(colored).copy()
    if label:
        cv2.putText(colored, label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return colored


def make_side_by_side(left, right, divider_px=4):
    """Concatenates two same-height BGR frames with a black divider."""
    h = left.shape[0]
    divider = np.zeros((h, divider_px, 3), dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def write_video(frames, out_path, fps):
    """Writes a list of BGR frames to an .mp4 file."""
    if not frames:
        print(f"  [WARNING] no frames to write for {out_path}")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(frame)

    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  → saved {len(frames)} frames  ({size_mb:.1f} MB)  {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-demo renderers
# ─────────────────────────────────────────────────────────────────────────────

def render_rgb_video(ep_grp, out_path, fps, instruction=""):
    """
    Renders a side-by-side RGB video:
        [agentview_rgb | eye_in_hand_rgb]
    """
    agentview   = ep_grp["obs/agentview_rgb"][()]       # (T, H, W, 3)
    eye_in_hand = ep_grp["obs/eye_in_hand_rgb"][()]     # (T, H, W, 3)
    T = len(agentview)

    frames = []
    for t in range(T):
        left  = prep_rgb(agentview[t],   label=f"agentview  t={t}")
        right = prep_rgb(eye_in_hand[t], label=f"eye_in_hand")
        row   = make_side_by_side(left, right)

        # instruction banner at top
        if instruction:
            banner = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, instruction[:90], (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            row = np.vstack([banner, row])

        frames.append(row)

    write_video(frames, out_path, fps)


def render_depth_video(ep_grp, out_path, fps, instruction=""):
    """
    Renders a side-by-side depth video (TURBO colormap):
        [agentview_depth | eye_in_hand_depth]
    """
    if "obs/agentview_depth" not in ep_grp:
        print(f"  [SKIP] no depth data found in {ep_grp.name}")
        return
    if "obs/eye_in_hand_depth" not in ep_grp:
        print(f"  [SKIP] no eye_in_hand_depth found in {ep_grp.name}")
        return

    agentview   = ep_grp["obs/agentview_depth"][()]     # (T, H, W, 1)
    eye_in_hand = ep_grp["obs/eye_in_hand_depth"][()]   # (T, H, W, 1)
    T = len(agentview)

    frames = []
    for t in range(T):
        left  = prep_depth(agentview[t],   label=f"agentview depth  t={t}")
        right = prep_depth(eye_in_hand[t], label=f"eye_in_hand depth")
        row   = make_side_by_side(left, right)

        if instruction:
            banner = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
            cv2.putText(banner, instruction[:90], (5, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            row = np.vstack([banner, row])

        frames.append(row)

    write_video(frames, out_path, fps)



# ─────────────────────────────────────────────────────────────────────────────
# Environment reconstruction helper
# ─────────────────────────────────────────────────────────────────────────────
 
def _build_env_from_ep(ep_grp, camera_height=128, camera_width=128):
    from libero.libero.envs import OffScreenRenderEnv   # ← use this
    import robosuite.utils.transform_utils as T

    bddl_content = ep_grp.attrs.get("bddl_content", None)
    model_xml    = ep_grp.attrs.get("model_file",   None)

    if bddl_content is None:
        raise ValueError(f"No bddl_content attr in {ep_grp.name}.")
    if model_xml is None:
        raise ValueError(f"No model_file attr in {ep_grp.name}.")

    fd, tmp_bddl_path = tempfile.mkstemp(suffix=".bddl", prefix="libero_replay_")
    os.close(fd)
    with open(tmp_bddl_path, "w") as f:
        f.write(bddl_content)

    env_args = {
        "bddl_file_name":  tmp_bddl_path,
        "camera_heights":  camera_height,
        "camera_widths":   camera_width,
        "camera_depths":   True,                    # needed for depth replay
        "camera_names":    ["robot0_eye_in_hand", "agentview"],
    }

    env = OffScreenRenderEnv(**env_args)
    seed = int(ep_grp.attrs.get("seed", 0))
    env.reset()
    env.reset_from_xml_string(model_xml)

    # Step 1: reset robot to populate buffers (joints go to neutral here)
    for robot in env.robots:
        robot._load_controller()
        robot.reset(deterministic=True)

    # Step 2: restore actual init_state AFTER robot.reset()
    env.sim.set_state_from_flattened(ep_grp.attrs["init_state"])
    env.sim.forward()

    # Step 3: read EEF directly from sim (not from stale robot buffers)
    for robot in env.robots:
        eef_site = robot.controller.eef_name
        site_id  = env.sim.model.site_name2id(eef_site)
        
        goal_pos = env.sim.data.site_xpos[site_id].copy()
        goal_mat = env.sim.data.site_xmat[site_id].reshape(3, 3).copy().astype(np.float64)

        robot.controller.goal_pos = goal_pos
        robot.controller.goal_ori = goal_mat

        # ← also sync gripper controller
        if hasattr(robot, "gripper") and hasattr(robot.gripper, "current_action"):
            # read current gripper qpos from sim and set as goal
            gripper_qpos = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy()
            print(f"gripper qpos from sim: {gripper_qpos}")
            
            # gripper controller goal is a scalar in [-1, 1]
            # map qpos to action space
            gripper_range = robot.gripper.dof
            robot.gripper.current_action = gripper_qpos[0]



    return env, tmp_bddl_path


# ─────────────────────────────────────────────────────────────────────────────
# 1. Replay RGB video
# ─────────────────────────────────────────────────────────────────────────────
 
def render_replay_rgb_video(
    ep_grp,
    out_path,
    fps,
    instruction="",
    camera_height=128,
    camera_width=128,
    divergence_threshold=0.01,
):
    """
    Replays recorded actions in a reconstructed LIBERO env and renders
    a side-by-side RGB video:
        [agentview_rgb | eye_in_hand_rgb]
 
    Also checks for physics divergence by comparing replayed sim states
    against the stored states at each step.
 
    Args:
        ep_grp               : h5py episode group (data/demo_N)
        out_path             : output .mp4 path
        fps                  : video frame rate
        instruction          : text to overlay as banner
        camera_height/width  : render resolution (can differ from stored)
        divergence_threshold : warn if state error exceeds this (metres)
    """
    actions = ep_grp["actions"][()]    # (T, 7)
    states  = ep_grp["states"][()]     # (T, state_dim)
    T       = len(actions)
 
    print(f"  [REPLAY RGB] T={T} steps, resolution={camera_height}×{camera_width}")
 
    env, tmp_bddl_path = _build_env_from_ep(ep_grp, camera_height, camera_width)
 
    frames         = []
    max_divergence = 0.0
 
    try:
        for t, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
 
            # ── divergence check ──────────────────────────────────────────────
            if t + 1 < len(states):
                replayed_state = env.sim.get_state().flatten()
                err = np.linalg.norm(states[t + 1] - replayed_state)
                max_divergence = max(max_divergence, err)
                if err > divergence_threshold:
                    print(f"    [DIVERGE] step {t:3d}  err={err:.4f}")
 
            # ── render ────────────────────────────────────────────────────────
            left  = prep_rgb(obs["agentview_image"],
                             label=f"agentview (replay) t={t}")
            right = prep_rgb(obs["robot0_eye_in_hand_image"],
                             label="eye_in_hand (replay)")
            row   = make_side_by_side(left, right)
 
            if instruction:
                banner = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
                cv2.putText(banner, instruction[:90], (5, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200),
                            1, cv2.LINE_AA)
                row = np.vstack([banner, row])
 
            frames.append(row)
 
    finally:
        env.close()
        os.unlink(tmp_bddl_path)
 
    print(f"  [REPLAY RGB] max physics divergence: {max_divergence:.6f}")
    write_video(frames, out_path, fps)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. Replay depth video
# ─────────────────────────────────────────────────────────────────────────────
 
def render_replay_depth_video(
    ep_grp,
    out_path,
    fps,
    instruction="",
    camera_height=128,
    camera_width=128,
    divergence_threshold=0.01,
):
    """
    Replays recorded actions in a reconstructed LIBERO env and renders
    a side-by-side TURBO-colorized depth video:
        [agentview_depth | eye_in_hand_depth]
 
    Args: same as render_replay_rgb_video
    """
    actions = ep_grp["actions"][()]    # (T, 7)
    states  = ep_grp["states"][()]     # (T, state_dim)
    T       = len(actions)
 
    print(f"  [REPLAY DEPTH] T={T} steps, resolution={camera_height}×{camera_width}")
 
    env, tmp_bddl_path = _build_env_from_ep(ep_grp, camera_height, camera_width)
 
    frames         = []
    max_divergence = 0.0
 
    try:
        for t, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
 
            # ── divergence check ──────────────────────────────────────────────
            if t + 1 < len(states):
                replayed_state = env.sim.get_state().flatten()
                err = np.linalg.norm(states[t + 1] - replayed_state)
                max_divergence = max(max_divergence, err)
                if err > divergence_threshold:
                    print(f"    [DIVERGE] step {t:3d}  err={err:.4f}")
 
            # ── render ────────────────────────────────────────────────────────
            left  = prep_depth(obs["agentview_depth"],
                               label=f"agentview depth (replay) t={t}")
            right = prep_depth(obs["robot0_eye_in_hand_depth"],
                               label="eye_in_hand depth (replay)")
            row   = make_side_by_side(left, right)
 
            if instruction:
                banner = np.zeros((24, row.shape[1], 3), dtype=np.uint8)
                cv2.putText(banner, instruction[:90], (5, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200),
                            1, cv2.LINE_AA)
                row = np.vstack([banner, row])
 
            frames.append(row)
 
    finally:
        env.close()
        os.unlink(tmp_bddl_path)
 
    print(f"  [REPLAY DEPTH] max physics divergence: {max_divergence:.6f}")
    write_video(frames, out_path, fps)
 

 # ─────────────────────────────────────────────────────────────────────────────
# Drop-in replacement for visualize_hdf5() that uses replay renderers
# ─────────────────────────────────────────────────────────────────────────────
 
def visualize_hdf5_replay(
    hdf5_path,
    demo_filter=None,
    out_dir=None,
    fps=10,
    camera_height=128,
    camera_width=128,
):
    """
    Same interface as visualize_hdf5() but renders by replaying actions
    in the simulator rather than reading stored images.
 
    Use this to:
      - verify stored actions reproduce the trajectory
      - re-render at a different resolution (set camera_height/width)
    """
    import json
    from pathlib import Path
 
    hdf5_path = str(hdf5_path)
    if out_dir is None:
        out_dir = str(Path(hdf5_path).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
 
    stem = Path(hdf5_path).stem
 
    with h5py.File(hdf5_path, "r") as f:
        try:
            problem_info = json.loads(f["data"].attrs["problem_info"])
            instruction  = problem_info.get("language_instruction", "")
        except Exception:
            instruction = ""
 
        demos = sorted(f["data"].keys())
        if demo_filter is not None:
            if demo_filter not in demos:
                raise ValueError(f"'{demo_filter}' not in file. Available: {demos}")
            demos = [demo_filter]
 
        print(f"\n[REPLAY] {hdf5_path}")
        print(f"  instruction : {instruction}")
        print(f"  demos       : {demos}")
        print(f"  resolution  : {camera_height}×{camera_width}")
        print(f"  output dir  : {out_dir}")
 
        for ep_name in demos:
            ep_grp    = f[f"data/{ep_name}"]
            T         = ep_grp.attrs.get("num_samples", "?")
            ep_instr  = ep_grp.attrs.get("instruction", instruction)
 
            print(f"\n── {ep_name}  (T={T}) ─────────────────────────────")
 
            # ── 1. replay RGB ─────────────────────────────────────────────────
            rgb_path = os.path.join(out_dir, f"{stem}_{ep_name}_replay_rgb.mp4")
            print(f"  [RGB]")
            render_replay_rgb_video(
                ep_grp, rgb_path, fps,
                instruction=ep_instr,
                camera_height=camera_height,
                camera_width=camera_width,
            )
 
            # ── 2. replay depth ───────────────────────────────────────────────
            depth_path = os.path.join(out_dir, f"{stem}_{ep_name}_replay_depth.mp4")
            print(f"  [DEPTH]")
            render_replay_depth_video(
                ep_grp, depth_path, fps,
                instruction=ep_instr,
                camera_height=camera_height,
                camera_width=camera_width,
            )
 
    print(f"\n[DONE] replay videos written to {out_dir}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def visualize_hdf5(hdf5_path, demo_filter=None, out_dir=None, fps=10):
    """
    Renders RGB and depth videos for each demo in an HDF5 file.

    Args:
        hdf5_path   : path to LIBERO-format HDF5
        demo_filter : if set (e.g. "demo_0"), only render that episode
        out_dir     : output directory (default: same folder as HDF5)
        fps         : video frame rate
    """
    hdf5_path = str(hdf5_path)
    if out_dir is None:
        out_dir = str(Path(hdf5_path).parent)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    stem = Path(hdf5_path).stem   # e.g. "Pick_the_closest_object_demo"

    with h5py.File(hdf5_path, "r") as f:
        # ── read instruction ──────────────────────────────────────────────────
        try:
            import json
            problem_info = json.loads(f["data"].attrs["problem_info"])
            instruction  = problem_info.get("language_instruction", "")
        except Exception:
            instruction = ""

        demos = sorted(f["data"].keys())
        if demo_filter is not None:
            if demo_filter not in demos:
                raise ValueError(f"'{demo_filter}' not found in {hdf5_path}. "
                                 f"Available: {demos}")
            demos = [demo_filter]

        print(f"\n[HDF5] {hdf5_path}")
        print(f"  instruction : {instruction}")
        print(f"  demos       : {demos}")
        print(f"  output dir  : {out_dir}")
        print(f"  fps         : {fps}")

        for ep_name in demos:
            ep_grp = f[f"data/{ep_name}"]
            T      = ep_grp.attrs.get("num_samples", "?")
            print(f"\n── {ep_name}  (T={T}) ─────────────────────────────")

            # ── 1. RGB video ──────────────────────────────────────────────────
            rgb_path = os.path.join(out_dir, f"{stem}_{ep_name}_rgb.mp4")
            print(f"  [RGB]")
            render_rgb_video(ep_grp, rgb_path, fps, instruction)

            # ── 2. Depth video ────────────────────────────────────────────────
            depth_path = os.path.join(out_dir, f"{stem}_{ep_name}_depth.mp4")
            print(f"  [DEPTH]")
            render_depth_video(ep_grp, depth_path, fps, instruction)

    print(f"\n[DONE] videos written to {out_dir}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--hdf5",  type=str, required=True,
#                         help="path to LIBERO-format HDF5 file")
#     parser.add_argument("--demo",  type=str, default=None,
#                         help="specific demo to render (e.g. demo_0). "
#                              "Default: render all demos")
#     parser.add_argument("--out",   type=str, default=None,
#                         help="output directory. Default: same folder as HDF5")
#     parser.add_argument("--fps",   type=int, default=10,
#                         help="video frame rate (default: 10)")
#     args = parser.parse_args()

#     visualize_hdf5(
#         hdf5_path=args.hdf5,
#         demo_filter=args.demo,
#         out_dir=args.out,
#         fps=args.fps,
#     )

import h5py
import hashlib
import re

hdf5_path = "/Hyperplane/shuijie/trajectory_data_hdf5/Pick_the_furtherest_object_and_place_in_the_bowl._demo.hdf5"

with h5py.File(hdf5_path, "r") as f:
    demos = sorted(f["data"].keys())
    print(f"Total demos: {len(demos)}\n")

    bddl_hashes = set()
    for ep in demos:
        ep_grp = f[f"data/{ep}"]

        bddl  = ep_grp.attrs.get("bddl_content", "MISSING")
        instr = ep_grp.attrs.get("instruction",  "MISSING")

        if bddl != "MISSING":
            # ── extract target object from :goal block ────────────────────
            goal_match = re.search(r":goal\s+\(And\s+\(On\s+(\S+)\s+(\S+)\)\)", bddl)
            obj_inst  = goal_match.group(1) if goal_match else "?"   # e.g. butter_0
            bowl_inst = goal_match.group(2) if goal_match else "?"   # e.g. white_bowl_0
            obj_type  = obj_inst.rsplit("_", 1)[0]
            bowl_type = bowl_inst.rsplit("_", 1)[0]

            # ── use full hash as fingerprint ──────────────────────────────
            bddl_hash = hashlib.md5(bddl.encode()).hexdigest()[:8]
            bddl_hashes.add(bddl_hash)
        else:
            obj_type = bowl_type = bddl_hash = "?"

        print(f"  {ep}: obj={obj_type:30s} bowl={bowl_type:20s} "
              f"bddl_hash={bddl_hash} | instr='{instr}'")

    print(f"\nUnique BDDLs: {len(bddl_hashes)} / {len(demos)}")
    if len(bddl_hashes) == len(demos):
        print("✅ All demos have different BDDLs")
    else:
        print("⚠️  Some demos share the same BDDL")


parser = argparse.ArgumentParser()
parser.add_argument("--hdf5",  type=str, default=hdf5_path,
                    help="path to LIBERO-format HDF5 file")
parser.add_argument("--demo",  type=str, default=None,
                    help="specific demo to render (e.g. demo_0). "
                            "Default: render all demos")
parser.add_argument("--out",   type=str, default=None,
                    help="output directory. Default: same folder as HDF5")
parser.add_argument("--fps",   type=int, default=10,
                    help="video frame rate (default: 10)")
args = parser.parse_args()

visualize_hdf5(
    hdf5_path=args.hdf5,
    demo_filter=args.demo,
    out_dir=args.out,
    fps=args.fps,
)


visualize_hdf5_replay(
    hdf5_path=args.hdf5,
    demo_filter=args.demo,
    out_dir=args.out,
    fps=args.fps,
    camera_height=256,
    camera_width=256,
)