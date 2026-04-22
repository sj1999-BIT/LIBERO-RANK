"""
keyframe_data.py

Two functions:
    save_keyframes(path, demos)   — write grounding-head training data to HDF5
    render_keyframes(path)        — read it back and display every keyframe

HDF5 layout
-----------
/demo_{i}/
    instruction         str
    state_transitions/
        transition_{j}/
            agentview_rgb       (H, W, 3)  uint8
            eye_in_hand_rgb     (H, W, 3)  uint8
            target_xyz          (3,)       float32
            target_role         str  — "pick" | "place"
            from_state          int
            to_state            int
"""

import h5py
import numpy as np
import cv2
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# State-transition → target role table
# ─────────────────────────────────────────────────────────────────────────────

# States 0-2 are approach/grasp phases → target is the pick object
# States 3-5 are transit/place phases  → target is the place bowl
_PICK_STATES  = {0, 1, 2}   # transitions FROM these states → pick target
_PLACE_STATES = {3, 4, 5}   # transitions FROM these states → place target

# All six transitions in the AutoGenPolicy state machine
TRANSITIONS = [
    (0, 1), (1, 2), (2, 3),   # approach + grasp
    (3, 4), (4, 5), (5, 6),   # transit + place
]


# ─────────────────────────────────────────────────────────────────────────────
# save_keyframes
# ─────────────────────────────────────────────────────────────────────────────

def save_keyframes(path: str, demos: list[dict]) -> None:
    """
    Save grounding-head training data to an HDF5 file.

    Parameters
    ----------
    path : str
        Destination .hdf5 file path.  Parent directories are created if needed.

    demos : list of dict
        Each dict must contain:
            instruction         str
            keyframes           list of 6 dicts, one per state transition,
                                each with keys:
                    agentview_rgb       np.ndarray  (H, W, 3) uint8
                    eye_in_hand_rgb     np.ndarray  (H, W, 3) uint8
                    pick_pos            np.ndarray  (3,) float32   — pick object XYZ
                    place_pos           np.ndarray  (3,) float32   — place bowl XYZ
                    from_state          int         — state the machine just left
                    to_state            int         — state just entered

    The function resolves which target XYZ to store (pick vs place) from
    from_state automatically using _PICK_STATES / _PLACE_STATES.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for demo_idx, demo in enumerate(demos):
            grp = f.create_group(f"demo_{demo_idx}")

            # store instruction as a scalar string dataset
            grp.create_dataset(
                "instruction",
                data=np.bytes_(demo["instruction"]),
            )

            tr_grp = grp.create_group("state_transitions")

            for tr_idx, kf in enumerate(demo["keyframes"]):
                from_s = int(kf["from_state"])
                to_s   = int(kf["to_state"])

                if from_s in _PICK_STATES:
                    target_xyz  = np.array(kf["pick_pos"],   dtype=np.float32)
                    target_role = "pick"
                elif from_s in _PLACE_STATES:
                    target_xyz  = np.array(kf["place_pos"],  dtype=np.float32)
                    target_role = "place"
                else:
                    raise ValueError(
                        f"from_state={from_s} not in pick or place state sets"
                    )

                kf_grp = tr_grp.create_group(f"transition_{tr_idx}")
                kf_grp.create_dataset(
                    "agentview_rgb",
                    data=np.array(kf["agentview_rgb"], dtype=np.uint8),
                    compression="gzip", compression_opts=4,
                )
                kf_grp.create_dataset(
                    "eye_in_hand_rgb",
                    data=np.array(kf["eye_in_hand_rgb"], dtype=np.uint8),
                    compression="gzip", compression_opts=4,
                )
                kf_grp.create_dataset("target_xyz",  data=target_xyz)
                kf_grp.create_dataset("from_state",  data=from_s)
                kf_grp.create_dataset("to_state",    data=to_s)
                kf_grp.create_dataset(
                    "target_role",
                    data=np.bytes_(target_role),
                )

    print(f"[save_keyframes] wrote {len(demos)} demos → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# render_keyframes
# ─────────────────────────────────────────────────────────────────────────────

def render_keyframes(path: str, delay_ms: int = 800) -> None:
    """
    Read a keyframe HDF5 file and display every frame with overlaid metadata.

    Press ESC or Q to quit early.  Any other key advances immediately.

    Parameters
    ----------
    path      : str   — path to the .hdf5 file written by save_keyframes
    delay_ms  : int   — milliseconds to pause between frames (default 800)
    """

    def _overlay(img: np.ndarray, lines: list[str]) -> np.ndarray:
        """Draw text lines onto a copy of img (expects RGB, returns RGB)."""
        out = img.copy()
        for i, line in enumerate(lines):
            cv2.putText(
                out, line,
                (6, 18 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out

    with h5py.File(path, "r") as f:
        demo_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[1]))

        for demo_key in demo_keys:
            demo_grp    = f[demo_key]
            instruction = demo_grp["instruction"][()].decode("utf-8")
            tr_grp      = demo_grp["state_transitions"]
            tr_keys     = sorted(tr_grp.keys(), key=lambda k: int(k.split("_")[1]))

            for tr_key in tr_keys:
                kf = tr_grp[tr_key]

                agentview    = kf["agentview_rgb"][()]       # (H, W, 3) RGB
                eye_in_hand  = kf["eye_in_hand_rgb"][()]
                target_xyz   = kf["target_xyz"][()]
                from_s       = int(kf["from_state"][()])
                to_s         = int(kf["to_state"][()])
                role         = kf["target_role"][()].decode("utf-8")

                xyz_str = f"xyz=({target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f})"
                meta_lines = [
                    f"{demo_key}  {tr_key}",
                    f"state {from_s}→{to_s}  [{role}]",
                    xyz_str,
                    instruction[:48],          # truncate long instructions
                ]

                # annotate both views
                av_disp  = _overlay(agentview,   meta_lines)
                eih_disp = _overlay(eye_in_hand, meta_lines)

                # resize eye-in-hand to match agentview height if needed
                if eih_disp.shape[0] != av_disp.shape[0]:
                    scale   = av_disp.shape[0] / eih_disp.shape[0]
                    new_w   = int(eih_disp.shape[1] * scale)
                    eih_disp = cv2.resize(eih_disp, (new_w, av_disp.shape[0]))

                # side-by-side panel, RGB → BGR for cv2
                panel = np.concatenate([av_disp, eih_disp], axis=1)
                panel_bgr = panel[..., ::-1]

                cv2.imshow("Keyframe Viewer", panel_bgr)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (27, ord("q")):       # ESC or Q → quit
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()
    print("[render_keyframes] done")


# ─────────────────────────────────────────────────────────────────────────────
# Integration sketch — how to plug into generate_trajectory
# ─────────────────────────────────────────────────────────────────────────────

def extract_keyframes_from_rollout(
    policy,
    obs_at_transitions: dict,   # keyed by (from_state, to_state) → obs snapshot
    pick_pos: np.ndarray,
    place_pos: np.ndarray,
) -> list[dict]:
    """
    Build the keyframes list expected by save_keyframes from the obs snapshots
    you capture during the rollout.

    In generate_trajectory, whenever _update_state() changes self.state,
    record the current obs into obs_at_transitions[(old_state, new_state)].
    Then pass that dict here after the episode.
    """
    keyframes = []
    for from_s, to_s in TRANSITIONS:
        obs = obs_at_transitions.get((from_s, to_s))
        if obs is None:
            continue    # transition didn't happen (e.g. episode too short)
        keyframes.append({
            "agentview_rgb":    obs["agentview_image"],
            "eye_in_hand_rgb":  obs["robot0_eye_in_hand_image"],
            "pick_pos":         pick_pos,
            "place_pos":        place_pos,
            "from_state":       from_s,
            "to_state":         to_s,
        })
    return keyframes


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test (runs without a real env)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os

    H, W = 256, 256
    rng  = np.random.default_rng(0)

    fake_demos = []
    for d in range(3):
        kfs = []
        for from_s, to_s in TRANSITIONS:
            kfs.append({
                "agentview_rgb":   rng.integers(0, 255, (H, W, 3), dtype=np.uint8),
                "eye_in_hand_rgb": rng.integers(0, 255, (H, W, 3), dtype=np.uint8),
                "pick_pos":        rng.random(3).astype(np.float32),
                "place_pos":       rng.random(3).astype(np.float32),
                "from_state":      from_s,
                "to_state":        to_s,
            })
        fake_demos.append({"instruction": f"Pick the closest object and place in the bowl. demo={d}", "keyframes": kfs})

    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
        tmp_path = tmp.name

    save_keyframes(tmp_path, fake_demos)
    print(f"Saved to {tmp_path}")

    render_keyframes(tmp_path, delay_ms=600)
    os.unlink(tmp_path)