"""
Single-Stage Data Collection
==============================
Collects a complete LIBERO-format HDF5 in one shot:
  - RGB images (agentview + eye_in_hand)
  - Depth images (agentview + eye_in_hand)
  - ee_states, gripper_states, joint_states
  - robot_states (manually built — OffScreenRenderEnv lacks get_robot_state_vector)
  - actions, states, rewards, dones

CAP_INDEX is applied at SAVE TIME (slice all arrays [CAP_INDEX:]) so that
collect arrays are always the same length during the rollout — no T mismatch.

robot_states assumption: ee_pos(3) + ee_ori(3) + gripper_qpos(2) = 8-dim.
If the reference dataset shows (T, 9), change the hstack below to add
gripper_qvel[:1] and set ROBOT_STATE_DIM = 9.
"""

import h5py
import json
import numpy as np
import os
from pathlib import Path
from random import randint

import robosuite.macros as macros
import robosuite.utils.transform_utils as T

from libero.libero.benchmark.rank_scripts.bddl_generator import (
    generate_random_rank_task_bddl,
    debug_log_print,
    get_obj_type,
    INSTRUCTION_TEMPLATES,
    OBJECT_POOL,
    BOWL_POOL,
)
from trajectory_generator import generate_trajectory, AutoGenPolicy

# ── tuneable constants ────────────────────────────────────────────────────────
CAP_INDEX       = 0      # drop first N steps (force sensor unstable)
N_DEMOS         = 3
HDF5_ROOT       = "/Hyperplane/shuijie/trajectory_data_hdf5"
ROBOT_STATE_DIM = 8      # change to 9 if you add gripper_qvel[:1]


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 writer
# ─────────────────────────────────────────────────────────────────────────────

def save_demos_to_hdf5(demos, hdf5_path, env_name, problem_name, env_kwargs):
    """
    Writes a complete LIBERO-format HDF5 from single-stage collected demos.

    Each demo dict must contain:
        actions           np.ndarray (T, 7)
        states            np.ndarray (T, 92)
        agentview_rgb     np.ndarray (T, H, W, 3)
        eye_in_hand_rgb   np.ndarray (T, H, W, 3)
        agentview_depth   np.ndarray (T, H, W, 1)
        eye_in_hand_depth np.ndarray (T, H, W, 1)
        ee_states         np.ndarray (T, 6)
        gripper_states    np.ndarray (T, 2)
        joint_states      np.ndarray (T, 7)
        robot_states      np.ndarray (T, 8 or 9)
        rewards           np.ndarray (T,)
        dones             np.ndarray (T,)
        model_xml         str
        init_state        np.ndarray (92,)
        instruction       str
        bddl_content      str
    """
    Path(hdf5_path).parent.mkdir(parents=True, exist_ok=True)

    

    with h5py.File(hdf5_path, "w") as f:
        grp = f.create_group("data")

        # ── top-level attrs ───────────────────────────────────────────────────
        grp.attrs["env_name"]                = env_name
        grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION
        grp.attrs["bddl_file_content"]       = demos[0]["bddl_content"]

        problem_info = {
            "problem_name":         problem_name,
            "language_instruction": demos[0]["instruction"],
            "domain_name":          "libero_rank",
        }
        grp.attrs["problem_info"] = json.dumps(problem_info)

        env_args = {
            "type":         1,
            "env_name":     env_name,
            "problem_name": problem_name,
            "env_kwargs":   env_kwargs,
        }
        grp.attrs["env_args"] = json.dumps(env_args)

        total_len = 0

        for i, demo in enumerate(demos):
            print(demo.keys())
            ep = grp.create_group(f"demo_{i}")

            # ── per-episode attrs ─────────────────────────────────────────────
            ep.attrs["model_file"]   = demo["model_xml"]
            ep.attrs["init_state"]   = demo["init_state"]
            ep.attrs["bddl_content"] = demo["bddl_content"]
            ep.attrs["num_samples"]  = len(demo["actions"])
            ep.attrs["instruction"]  = demo["instruction"]   # ← was missing
            ep.attrs["seed"]         = demo["seed"]

            # ── obs group ─────────────────────────────────────────────────────
            obs_grp = ep.create_group("obs")
            obs_grp.create_dataset("agentview_rgb",     data=demo["agentview_rgb"])
            obs_grp.create_dataset("eye_in_hand_rgb",   data=demo["eye_in_hand_rgb"])
            obs_grp.create_dataset("agentview_depth",   data=demo["agentview_depth"])
            obs_grp.create_dataset("eye_in_hand_depth", data=demo["eye_in_hand_depth"])
            obs_grp.create_dataset("ee_states",         data=demo["ee_states"])
            obs_grp.create_dataset("ee_pos",            data=demo["ee_states"][:, :3])
            obs_grp.create_dataset("ee_ori",            data=demo["ee_states"][:, 3:])
            obs_grp.create_dataset("gripper_states",    data=demo["gripper_states"])
            obs_grp.create_dataset("joint_states",      data=demo["joint_states"])

            # ── top-level datasets ────────────────────────────────────────────
            ep.create_dataset("actions",      data=demo["actions"])
            ep.create_dataset("states",       data=demo["states"])
            ep.create_dataset("robot_states", data=demo["robot_states"])
            ep.create_dataset("rewards",      data=demo["rewards"])
            ep.create_dataset("dones",        data=demo["dones"])

            total_len += len(demo["actions"])

            print(f"  [demo_{i}] T={len(demo['actions'])} | "
                  f"rgb={demo['agentview_rgb'].shape} | "
                  f"depth={demo['agentview_depth'].shape} | "
                  f"robot_states={demo['robot_states'].shape}")

        grp.attrs["num_demos"] = len(demos)
        grp.attrs["total"]     = total_len

    print(f"\n[SAVED] {len(demos)} demos → {hdf5_path}  (total_transitions={total_len})")


# ─────────────────────────────────────────────────────────────────────────────
# Extract + validate + apply CAP_INDEX
# ─────────────────────────────────────────────────────────────────────────────

def process_demo(demo):
    """
    Applies CAP_INDEX slice to all time-indexed arrays and runs sanity checks.
    Returns processed demo dict, or raises AssertionError on failure.
    """
    c = CAP_INDEX

    processed = {
        # ── apply CAP_INDEX ───────────────────────────────────────────────────
        "actions":           demo["actions"][c:],
        "states":            demo["states"][c:],
        "agentview_rgb":     demo["agentview_rgb"][c:],
        "eye_in_hand_rgb":   demo["eye_in_hand_rgb"][c:],
        "agentview_depth":   demo["agentview_depth"][c:],
        "eye_in_hand_depth": demo["eye_in_hand_depth"][c:],
        "ee_states":         demo["ee_states"][c:],
        "gripper_states":    demo["gripper_states"][c:],
        "joint_states":      demo["joint_states"][c:],
        "robot_states":      demo["robot_states"][c:],
        "rewards":           demo["rewards"][c:],
        "dones":             demo["dones"][c:],
        # ── scalars / strings (no slicing) ───────────────────────────────────
        "model_xml":     demo["model_xml"],
        "init_state":    demo["init_state"],
        "instruction":   demo["instruction"],
        "bddl_content":  demo["bddl_content"],
        "seed":          demo["seed"],   # ← add this
    }

    T_steps = len(processed["actions"])

    # ── all time axes must match ──────────────────────────────────────────────
    for key in ["states", "agentview_rgb", "eye_in_hand_rgb",
                "agentview_depth", "eye_in_hand_depth",
                "ee_states", "gripper_states", "joint_states",
                "robot_states", "rewards", "dones"]:
        assert len(processed[key]) == T_steps, \
            f"T mismatch: actions={T_steps}, {key}={len(processed[key])}"

    # ── shape checks ──────────────────────────────────────────────────────────
    assert processed["actions"].shape[1]       == 7,              f"actions dim: {processed['actions'].shape}"
    assert processed["ee_states"].shape[1]     == 6,              f"ee_states dim: {processed['ee_states'].shape}"
    assert processed["gripper_states"].shape[1]== 2,              f"gripper_states dim: {processed['gripper_states'].shape}"
    assert processed["joint_states"].shape[1]  == 7,              f"joint_states dim: {processed['joint_states'].shape}"
    assert processed["robot_states"].shape[1]  == ROBOT_STATE_DIM,f"robot_states dim: {processed['robot_states'].shape}"
    assert processed["agentview_rgb"].ndim     == 4,              f"agentview_rgb not (T,H,W,C)"
    assert processed["agentview_depth"].ndim   == 4,              f"agentview_depth not (T,H,W,1)"

    # ── action bounds ─────────────────────────────────────────────────────────
    a_min = processed["actions"].min()
    a_max = processed["actions"].max()
    assert a_min >= -1.0 and a_max <= 1.0, \
        f"actions out of [-1,1]: min={a_min:.3f} max={a_max:.3f}"

    return processed


# ─────────────────────────────────────────────────────────────────────────────
# generate_trajectory post-processor
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_data(raw):
    """
    Converts the raw dict returned by generate_trajectory into the full
    demo dict expected by process_demo / save_demos_to_hdf5.

    robot_states: ee_pos(3) + ee_ori(3) + gripper_qpos(2) = 8-dim
    If you need 9-dim, add gripper_qvel[:1] to the hstack and set
    ROBOT_STATE_DIM = 9 at the top of this file.
    """
    T = len(raw["actions"])

    # build rewards and dones (sparse: 0 everywhere, 1 at last step)
    rewards = np.zeros(T, dtype=np.uint8)
    dones   = np.zeros(T, dtype=np.uint8)
    rewards[-1] = 1
    dones[-1]   = 1

    return {
        "actions":           raw["actions"],
        "states":            raw["states"],
        "agentview_rgb":     raw["agentview_rgb"],
        "eye_in_hand_rgb":   raw["eye_in_hand_rgb"],
        "agentview_depth":   raw["agentview_depth"],
        "eye_in_hand_depth": raw["eye_in_hand_depth"],
        "ee_states":         raw["ee_states"],
        "gripper_states":    raw["gripper_states"],
        "joint_states":      raw["joint_states"],
        "robot_states":      raw["robot_states"],   # already built in generate_trajectory
        "rewards":           rewards,
        "dones":             dones,
        "model_xml":         raw["model_xml"],
        "init_state":        raw["init_state"],
        "instruction":       raw["instruction"],
        "bddl_content":      raw["bddl_content"],
        "seed":              raw["seed"],   # ← add this
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    BASE_ENV_KWARGS = {
        "robots":         ["Panda"],
        "camera_heights": 128,
        "camera_widths":  128,
        "use_camera_obs": True,
        "camera_names":   ["robot0_eye_in_hand", "agentview"],
        "camera_depths":  True,
        "reward_shaping": True,
        "control_freq":   20,
    }

    # for testing_object_type in OBJECT_POOL:
    #     for testing_bowl_type in BOWL_POOL:
    for testing_instruction in INSTRUCTION_TEMPLATES:

        safe_instr = testing_instruction.strip().replace(" ", "_")[:60]
        hdf5_path  = os.path.join(
            HDF5_ROOT,
            # testing_object_type,
            # testing_bowl_type,
            f"{safe_instr}_demo.hdf5",
        )

        if os.path.exists(hdf5_path):
            print(f"[SKIP] {hdf5_path}")
            continue

        print(f"\n{'='*60}")
        # print(f"OBJ={testing_object_type} | BOWL={testing_bowl_type}")
        print(f"INSTR={testing_instruction}")
        print(f"{'='*60}")

        demos   = []
        attempt = 0

        while len(demos) < N_DEMOS:
            attempt += 1
            print(f"  [attempt {attempt}] collected={len(demos)}/{N_DEMOS}")

            try:
                _, _, demo_data_list = generate_trajectory(
                    instruction=testing_instruction,
                    policy=AutoGenPolicy(is_debugging=False),
                    camera_heights=BASE_ENV_KWARGS["camera_heights"],
                    camera_widths=BASE_ENV_KWARGS["camera_widths"],   
                    object_type=None,
                    bow_type=None,
                    num_objects=10,
                    trajectory_len=300,
                    save_bddl=True,
                    render_video=False,
                    is_log_printed=False,
                    env_grid_len=1,
                )
            except Exception as e:
                print(f"  [WARNING] generate_trajectory raised: {e}")
                continue

            if not demo_data_list:
                print(f"  [WARNING] no successful demos this attempt")
                continue

            for raw in demo_data_list:
                try:
                    demo = build_demo_data(raw)
                    demo = process_demo(demo)   # apply CAP_INDEX + checks
                    demos.append(demo)
                    print(f"  → accepted demo, T={len(demo['actions'])}, "
                            f"total={len(demos)}/{N_DEMOS}")
                except AssertionError as e:
                    print(f"  [WARNING] sanity check failed: {e}")
                    continue

        save_demos_to_hdf5(
            demos=demos[:N_DEMOS],
            hdf5_path=hdf5_path,
            env_name="LiberoRank",
            problem_name=testing_instruction,
            env_kwargs=BASE_ENV_KWARGS,
        )
        break