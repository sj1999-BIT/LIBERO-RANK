"""
Since all tasks in LIBERO_RANK are simple pick and place, in theory we could generate the correct trajectory to
complete all the tasks if we have the location of the objects.

1. we need a naive trajectory generator that does the following.
    1. move the end effector to the top of the target object
    2. grab the object firmly and move up
    3. move to the target bowl instance directly above
    4. lower to appropriate height before opening the gripper.

2. For each object, the way its placed its always the same. Hence the idea is that given an object,
   the target gripping point should remain consistent.
    1. for each object, design a fixed gripping point (assume the object global pos as the centre).
    2. for each bowl, design a fixed dropping point (assume the bowl global pos as the centre).

State machine:
    0 → align XY above pick target, gripper open
    1 → lower Z to pick height
    2 → close gripper until velocity stalls (object grabbed)
    3 → lift arm while moving XY toward place target (avoids singularity on retraction)
    4 → align XY above place target at clearance height
    5 → lower Z to place height
    6 → open gripper, retract

Per-object offset tables
    OBJECT_PICK_PARAMS   [x_off, y_off, height]
        x/y: lateral offset added to the observed object pos to reach the
             correct gripper approach point (e.g. ramekin lip offset)
        height: z above table surface at which the gripper closes

    OBJECT_PLACE_OFFSET  [x_off, y_off, z_off]
        Correction added to place_pos to compensate for how the object sits
        in the gripper after grasping.  For centred grasps this is [0,0,0];
        for off-centre grasps (ramekin) it corrects the landing position so
        the object ends up centred over the target bowl.

Trajectory quality checks (all 4 must pass; failed attempts are retried):
    CHECK 1  reward >= 1.0 at episode end
    CHECK 2  state-5 Z reached target without stalling (gripper not pressing
             object into bowl).  Z-stall is detected when Z stops changing for
             Z_STALL_STEPS_MAX consecutive steps before reaching the target.
    CHECK 3  object did not stick to the gripper after opening — detected by:
             (a) [legacy] watching for the object Z rising more than OBJECT_STUCK_Z_RISE
                 once retraction begins (catches high-Z drop scenarios).
             (b) [new]    after done=True at state 6, running POST_DONE_EXTEND_STEPS
                 extra steps and checking whether the object shifts in Y by more
                 than OBJECT_STUCK_Y_SHIFT (catches low-Z drops where Z doesn't move).
    CHECK 4  state-machine did not time out (object was reachable).

Command
nohup /Hyperplane/conda/envs/libero/bin/python /root/code/LIBERO-RANK/trajectory_generator.py > /root/code/LIBERO-RANK/traj_gen.log 2>&1 &
"""

import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# os.environ["MKL_THREADING_LAYER"] = "GNU"
libero_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
print(f"libero path: {libero_path}")

import sys
sys.path.insert(0, libero_path)

from libero.libero.benchmark.rank_scripts.bddl_generator import (
    generate_random_rank_task_bddl, debug_log_print, get_obj_type,
    OBJECT_POOL, BOWL_POOL
)

from random import randint
import numpy as np
import cv2
from tqdm import tqdm
from libero.libero.envs import OffScreenRenderEnv
from trajectory_logger import TrajectoryLogger

import robosuite.utils.transform_utils as T


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def prep_for_display(img, instruction=None, lineLen=30):
    # only perform bgr to rgb if its 3 channels
    if img.shape[-1] == 3:
        img = img[..., ::-1]
    img = np.flipud(img).copy()
    if instruction is not None:
        words = instruction.split()
        lines, line = [], []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > lineLen:
                lines.append(" ".join(line))
                line = []
        if line:
            lines.append(" ".join(line))
        for i, text in enumerate(lines):
            cv2.putText(img, text, (5, 15 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def create_title(cur_instruction):
    return "_".join(cur_instruction.split(" ")[:-1])


# ──────────────────────────────────────────────────────────────────────────────
# Policy base class
# ──────────────────────────────────────────────────────────────────────────────

class Policy:
    def __init__(self, input_state_dict_keys: list, is_debugging: bool = False):
        self.input_state_dict_keys = input_state_dict_keys
        self.is_debugging = is_debugging

    def check_input_state_dict_keys(self, state_dict: dict):
        return all(k in state_dict for k in self.input_state_dict_keys)

    def _generate_action(self, state_dict):
        raise NotImplementedError(
            f"Need to implement _generate_action for {self.__class__.__name__}"
        )

    def get_action(self, state_dict):
        if not self.check_input_state_dict_keys(state_dict):
            absent = [k for k in self.input_state_dict_keys if k not in state_dict]
            raise ValueError(
                f"{self.__class__.__name__} missing required keys: {absent}"
            )
        return self._generate_action(state_dict)

    def reset(self):
        raise NotImplementedError(
            f"Need to implement reset for {self.__class__.__name__}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# ── gripper constants ─────────────────────────────────────────────────────────
GRIPPER_OPEN        = -1.0
GRIPPER_CLOSE       =  1.0
GRIP_HOLD_STEPS     = 10       # steps to dwell while closing before checking stall
GRIPPER_LENGTH      = 0.0

# Fixed clearance height above the table used during transit.
# Raised to 0.40 to safely clear tallest object (moka_pot ~0.14m)
# plus grasped item height and gripper body.
TRANSIT_CLEARANCE   = 0.40     # metres above TABLE_HEIGHT

# State-1 Z threshold (Bug 1 fix):
# The P-controller (action = delta * 10) converges asymptotically; 0.005 is
# too tight and causes 60+ crawling steps because the small residual action is
# absorbed by joint damping.  0.015 exits promptly while still landing the
# gripper close enough to the object for a reliable grasp.
PICK_Z_THRESHOLD    = 0.015

# ── trajectory quality check constants ───────────────────────────────────────
# Check 2: Z-stall detection in state 5
Z_STALL_THRESHOLD   = 0.002    # metres per step — below this counts as stalled
Z_STALL_STEPS_MAX   = 8        # consecutive stalled steps before forcing 5→6

# Check 3a (legacy): Object-stuck detection via Z-rise after gripper opens
OBJECT_STUCK_Z_RISE  = 0.03   # metres — if object rises more than this, it's stuck

# Check 3b (new): Object-stuck detection via Y-shift after done at state 6
# Catches low-Z drops where the object doesn't rise but is dragged sideways
# when the gripper fingers retract.
OBJECT_STUCK_Y_SHIFT    = 0.02  # metres — Y shift that signals still gripped
POST_DONE_EXTEND_STEPS  = 10    # extra steps run after done=True to observe Y

# Check 4: per-state step timeout
STATE_TIMEOUT_STEPS = {
    0: 100,   # XY alignment — should converge in <15 steps normally
    1: 100,   # Z lowering — same
    4: 100,   # XY alignment to place target
    5: 100,   # Z lowering to place height (already has stall guard)
}

# Retry budget per combination
MAX_RETRIES         = 50

# Height of tabletop for manipulation
TABLE_HEIGHT = 0.9


# ──────────────────────────────────────────────────────────────────────────────
# Object / bowl parameter tables
# ──────────────────────────────────────────────────────────────────────────────

INSTRUCTION_TEMPLATES = [
    # egocentric pick tasks: many same object + 1 bowl
    "Pick the closest object and place in the bowl.",
    "Pick the furtherest object and place in the bowl.",
    # egocentric place tasks: 1 object + many bowls
    "Pick the object and place in the closest bowl.",
    "Pick the object and place in the furtherest bowl.",
    # allocentric pick: many same object + 1 bowl
    "Pick the object closest to the bowl and place in the bowl.",
    "Pick the object furtherest to the bowl and place in the bowl.",
    # allocentric place: 1 object + many bowls
    "Pick the object and place in the bowl closest to it.",
    "Pick the object and place in the bowl furtherest from it.",
    # pick by feature: many different object + 1 bowl
    "Pick the largest object and place in the bowl.",
    "Pick the smallest object and place in the bowl.",
    # place by feature: 1 object + different bowl
    "Pick the object and place in the largest bowl.",
    "Pick the object and place in the smallest bowl.",
    # middle pick: 3 different object with col restriction + one bowl
    "Pick the object in the middle and place in the bowl."
    "Pick the object and place in the bowl in the middle."
]


# OBJECT_PICK_PARAMS  [x_offset, y_offset, height]
#   x/y: lateral offset from the observed object centre to the gripper
#        approach point.
#   height: z above the table surface at which the gripper closes.
OBJECT_PICK_PARAMS = {
    "milk":                          np.array([0.0,  0.0,  0.09]),
    "moka_pot":                      np.array([0.0,  0.0,  0.14]),
    "glazed_rim_porcelain_ramekin":  np.array([0.0,  0.03, 0.02]),
    "tomato_sauce":                  np.array([0.0,  0.0,  0.03]),
    "alphabet_soup":                 np.array([0.0,  0.0,  0.03]),
    "butter":                        np.array([0.0,  0.0,  0.0]),
    "ketchup":                       np.array([0.0,  0.0,  0.11]),
    "orange_juice":                  np.array([0.0,  0.0,  0.10]),
}


# OBJECT_PLACE_OFFSET  [x_offset, y_offset, z_offset]
#   Added to place_pos to compensate for how the object sits in the gripper
#   after grasping so the object lands centred over the target bowl.
OBJECT_PLACE_OFFSET = {
    "milk":                          np.array([0.0,   0.0,  0.03]),
    "moka_pot":                      np.array([0.0,   0.0,  0.08]),
    "glazed_rim_porcelain_ramekin":  np.array([0.0,  0.03, 0.01]),
    "tomato_sauce":                  np.array([0.0,   0.0,  0.01]),
    "alphabet_soup":                 np.array([0.0,   0.0,  0.01]),
    "butter":                        np.array([0.0,   0.0,  0.01]),
    "ketchup":                       np.array([0.0,   0.0,  0.05]),
    "orange_juice":                  np.array([0.0,   0.0,  0.04]),
}

BOWL_PLACE_POS = {
    "white_bowl":       np.array([0.0, 0.0, 0.03]),
    "akita_black_bowl": np.array([0.0, 0.0, 0.03]),
    "plate":            np.array([0.0, 0.0, 0.03]),
}


# ──────────────────────────────────────────────────────────────────────────────
# AutoGenPolicy
# ──────────────────────────────────────────────────────────────────────────────

class AutoGenPolicy(Policy):
    """
    7-state pick-and-place state machine.

    States
    ------
    0  Align XY to pick position, gripper open, hold Z above objects
    1  Lower Z to pick height
    2  Close gripper until velocity stalls → object grabbed
    3  Lift Z to TRANSIT_CLEARANCE *while* holding XY (diagonal arc avoids
       kinematic singularity on large-X retraction)
    4  Align XY precisely over place target at clearance height
    5  Lower Z to place height
    6  Open gripper, retract to neutral Z

    Quality flags (read by the caller after the episode)
    ----------------------------------------------------
    z_stalled    : bool — state-5 exited due to Z freeze (Check 2)
    unreachable  : bool — a state timed out (Check 4)
    """

    def __init__(self, is_debugging: bool = False):
        keys = [
            "pick_pos", "place_pos", "object_height",
            "cur_gripper_pos", "original_gripper_pos",
            "gripper_velocity",
        ]
        super().__init__(keys, is_debugging)
        self.state             = 0
        self.gripper_prev_vel  = 0.0
        self.close_steps       = 0
        self.open_steps        = 10
        self.remaining_open_steps = self.open_steps

        # Check 2: Z-stall detection
        self.z_stall_steps = 0
        self.prev_z        = None
        self.z_stalled     = False

        # Check 4: per-state timeout
        self.state_step_counter = 0
        self.unreachable        = False

    # ── helpers ───────────────────────────────────────────────────────────────

    def _transit_height(self):
        return TABLE_HEIGHT + GRIPPER_LENGTH + TRANSIT_CLEARANCE

    def _is_xy_reached(self, cur_pos, target_pos, threshold=0.025):
        return all(abs(cur_pos[i] - target_pos[i]) <= threshold for i in range(2))

    def _is_z_reached(self, cur_z, target_z, threshold=0.025):
        return abs(cur_z - target_z) <= threshold

    # ── state transitions ─────────────────────────────────────────────────────

    def _update_state(self, state_dict):
        cur  = state_dict["cur_gripper_pos"]
        pick = state_dict["pick_pos"]
        plc  = state_dict["place_pos"]
        th   = self._transit_height()

        self.state_step_counter += 1

        # ── 0: XY aligned to pick target ─────────────────────────────────────
        if self.state == 0:
            if self._is_xy_reached(cur, pick):
                debug_log_print("AutoGenPolicy", "0→1: XY aligned to pick", self.is_debugging)
                self.state = 1
                self.state_step_counter = 0
                return

        # ── 1: Z lowered to pick height ───────────────────────────────────────
        if self.state == 1:
            pick_z = TABLE_HEIGHT + GRIPPER_LENGTH + state_dict["object_height"]
            if self._is_z_reached(cur[2], pick_z, threshold=PICK_Z_THRESHOLD):
                debug_log_print("AutoGenPolicy", "1→2: at pick height", self.is_debugging)
                self.state = 2
                self.state_step_counter = 0
                return

        # ── 2: gripper stalls → object grabbed ───────────────────────────────
        if self.state == 2:
            self.close_steps += 1
            if self.close_steps >= GRIP_HOLD_STEPS:
                vel = state_dict["gripper_velocity"][1]
                if vel < self.gripper_prev_vel:
                    debug_log_print("AutoGenPolicy", "2→3: gripper stalled, object grabbed", self.is_debugging)
                    self.close_steps = 0
                    self.state = 3
                    self.state_step_counter = 0
                    return
                self.gripper_prev_vel = vel

        # ── 3: reached transit height → start XY alignment to place ──────────
        if self.state == 3:
            if self._is_z_reached(cur[2], th, threshold=0.02):
                debug_log_print("AutoGenPolicy", "3→4: at transit height, moving to place XY", self.is_debugging)
                self.state = 4
                self.state_step_counter = 0
                return

        # ── 4: XY aligned to place target ────────────────────────────────────
        if self.state == 4:
            if self._is_xy_reached(cur, plc):
                debug_log_print("AutoGenPolicy", "4→5: XY aligned to place target", self.is_debugging)
                self.state = 5
                self.state_step_counter = 0
                # Reset Z-stall tracker fresh on entering state 5
                self.z_stall_steps = 0
                self.prev_z        = None
                return

        # ── 5: Z lowered to place height ─────────────────────────────────────
        # CHECK 2: Z-stall guard — if Z freezes before reaching the target the
        # gripper is pressing the object into the bowl.  Force 5→6 and flag it.
        if self.state == 5:
            if self.prev_z is not None:
                z_delta = abs(cur[2] - self.prev_z)
                if z_delta < Z_STALL_THRESHOLD:
                    self.z_stall_steps += 1
                else:
                    self.z_stall_steps = 0

            self.prev_z = cur[2]

            if self.z_stall_steps >= Z_STALL_STEPS_MAX:
                debug_log_print(
                    "AutoGenPolicy",
                    f"5→6 [STALL]: Z frozen for {self.z_stall_steps} steps "
                    f"(cur_z={cur[2]:.4f}, target_z={plc[2]:.4f}) — forcing place",
                    self.is_debugging,
                )
                self.z_stalled     = True
                self.z_stall_steps = 0
                self.state         = 6
                return

            if self._is_z_reached(cur[2], plc[2], threshold=0.02):
                debug_log_print("AutoGenPolicy", "5→6: at place height, opening gripper", self.is_debugging)
                self.state = 6
                self.state_step_counter = 0
            return

        # ── 6: terminal state — no further transitions needed ─────────────────

        # ── Check 4: state timeout ────────────────────────────────────────────
        timeout = STATE_TIMEOUT_STEPS.get(self.state)
        if timeout is not None and self.state_step_counter >= timeout:
            debug_log_print(
                "AutoGenPolicy",
                f"CHECK 4 FAILED: state {self.state} timed out after "
                f"{self.state_step_counter} steps — object unreachable",
                self.is_debugging,
            )
            self.unreachable = True

    # ── action generation ─────────────────────────────────────────────────────

    def _generate_action(self, state_dict):
        self._update_state(state_dict)

        cur  = state_dict["cur_gripper_pos"]
        pick = state_dict["pick_pos"]
        plc  = state_dict["place_pos"]
        oh   = state_dict["object_height"]
        th   = self._transit_height()

        action = np.zeros(7)

        # ── 0: align XY to pick, hold Z above objects ─────────────────────────
        if self.state == 0:
            action[:2] = pick[:2] - cur[:2]
            action[2]  = -min(0.0, cur[2] - (TABLE_HEIGHT + oh + 0.05))
            action[6]  = GRIPPER_OPEN
            debug_log_print("AutoGenPolicy",
                f"state=0 | Δxy={action[:2]} | Δz={action[2]:.4f}", self.is_debugging)

        # ── 1: lower Z to pick height ─────────────────────────────────────────
        elif self.state == 1:
            pick_z     = TABLE_HEIGHT + GRIPPER_LENGTH + oh
            action[:2] = pick[:2] - cur[:2]
            action[2]  = pick_z - cur[2]
            action[6]  = GRIPPER_OPEN
            debug_log_print("AutoGenPolicy",
                f"state=1 | Δz={action[2]:.4f} | target_z={pick_z:.4f}", self.is_debugging)

        # ── 2: close gripper ──────────────────────────────────────────────────
        elif self.state == 2:
            action[6] = GRIPPER_CLOSE
            debug_log_print("AutoGenPolicy",
                f"state=2 | closing gripper | vel={state_dict['gripper_velocity']}", self.is_debugging)

        # ── 3: DIAGONAL ARC — lift Z while holding XY ─────────────────────────
        elif self.state == 3:
            action[2]  = th - cur[2]
            action[:2] = 0.0
            action[6]  = GRIPPER_CLOSE
            debug_log_print("AutoGenPolicy",
                f"state=3 | lifting | Δz={action[2]:.4f} | target_th={th:.4f}", self.is_debugging)

        # ── 4: align XY over place target at transit height ───────────────────
        elif self.state == 4:
            action[:2] = plc[:2] - cur[:2]
            action[2]  = -min(0.0, th - cur[2])
            action[6]  = GRIPPER_CLOSE
            debug_log_print("AutoGenPolicy",
                f"state=4 | aligning XY | Δxy={action[:2]}", self.is_debugging)

        # ── 5: lower Z to place height ────────────────────────────────────────
        elif self.state == 5:
            action[:2] = plc[:2] - cur[:2]
            action[2]  = plc[2] - cur[2]
            action[6]  = GRIPPER_CLOSE
            debug_log_print("AutoGenPolicy",
                f"state=5 | descending | Δz={action[2]:.4f} | target_z={plc[2]:.4f}", self.is_debugging)

        # ── 6: open gripper, retract upward ───────────────────────────────────
        elif self.state == 6:
            action[6]  = GRIPPER_OPEN
            self.open_steps -= 1
            if self.open_steps <= 0:
                action[2] = th - cur[2]
            debug_log_print("AutoGenPolicy",
                "state=6 | opening gripper & retracting", self.is_debugging)

        action[:3] = np.clip(action[:3] * 10, -1, 1)
        return action

    def reset(self):
        self.state             = 0
        self.gripper_prev_vel  = 0.0
        self.close_steps       = 0
        self.open_steps        = 10
        self.remaining_open_steps = self.open_steps
        # Check 2 trackers
        self.z_stall_steps     = 0
        self.prev_z            = None
        self.z_stalled         = False
        # Check 4 trackers
        self.state_step_counter = 0
        self.unreachable        = False


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_trajectory(
    instruction,
    policy: Policy,
    camera_heights=256,
    camera_widths=256,
    object_type=None,
    bow_type=None,
    num_objects=10,
    trajectory_len=100,
    save_bddl=True,
    output_path=None,
    env_grid_len=1,
    render_video=True,
    is_save_video=False,
    save_video_path=None,
    is_log_printed=False,
    logger: TrajectoryLogger = None,
):
    total_env = env_grid_len * env_grid_len
    total_trajectory = []
    results = []
    data_collected = []

    for cur_env in range(total_env):

        # ── retry loop: keep trying until all checks pass or budget runs out ──
        for attempt in range(1, MAX_RETRIES + 1):
            policy.reset()

            result = generate_random_rank_task_bddl(
                language=instruction,
                object_type=object_type,
                bow_type=bow_type,
                num_objects=num_objects,
                save_bddl=save_bddl,
                output_path=output_path,
                is_debugging=is_log_printed,
            )

            actual_instruction = result["resolved_language"]
            target_pick_label  = result["target_object"]
            target_place_label = result["target_place"]

            actual_object_type = get_obj_type(target_pick_label)
            actual_bow_type    = get_obj_type(target_place_label)

            env = OffScreenRenderEnv(
                bddl_file_name=result["bddl_path"],
                robots=["Panda"],
                camera_heights=camera_heights,
                camera_widths=camera_widths,
                use_camera_obs=True,
                camera_names=["robot0_eye_in_hand", "agentview"],
                camera_depths=True,
            )

            seed = randint(0, 1000)
            env.seed(seed)
            obs = env.reset()

            # ── DIM CHECK: available obs keys ──────────────────────────────
            print("\n[DIM CHECK] Available obs keys:")
            for k, v in obs.items():
                v_arr = np.array(v)
                print(f"  obs['{k}'] shape: {v_arr.shape}, dtype: {v_arr.dtype}")
            print()

            params    = OBJECT_PICK_PARAMS[actual_object_type]
            pick_pos  = obs[f"{target_pick_label}_pos"] + np.array([params[0], params[1], 0.0])
            place_pos = (obs[f"{target_place_label}_pos"]
                         + BOWL_PLACE_POS[actual_bow_type]
                         + OBJECT_PLACE_OFFSET[actual_object_type])

            original_gripper_pos = obs["robot0_eef_pos"]
            frames  = []
            success = False

            # ── Check 3a: legacy Z-rise tracking ─────────────────────────────
            placed_object_z = None

            # ── Check 3b: Y-shift tracking ────────────────────────────────────
            # Captures object Y at state-6 entry, then after done=True we run
            # POST_DONE_EXTEND_STEPS extra env steps to see if the object drifts
            # in Y (indicating the gripper dragged it when opening).
            placed_object_y   = None
            object_stuck      = False
            in_post_done      = False   # True once we enter the extension window
            post_done_counter = 0       # counts extension steps

            # ── Data collection arrays ────────────────────────────────────────
            model_xml_initial  = env.sim.model.get_xml()
            init_state         = env.sim.get_state().flatten()
            all_states         = []
            all_actions        = []
            all_robot_states   = []
            all_ee_states      = []
            all_gripper_states = []
            all_joint_states   = []
            all_rewards        = []
            agentview_images   = []
            eye_in_hand_images = []
            agentview_depths   = []
            eye_in_hand_depths = []

            CAP_INDEX = 0

            # ── Main rollout loop ─────────────────────────────────────────────
            for step in range(trajectory_len + POST_DONE_EXTEND_STEPS):
                cur_gripper_pos  = obs["robot0_eef_pos"]
                input_state_dict = {
                    "pick_pos":             pick_pos,
                    "place_pos":            place_pos,
                    "cur_gripper_pos":      cur_gripper_pos,
                    "original_gripper_pos": original_gripper_pos,
                    "object_height":        params[2],
                    "gripper_velocity":     obs["robot0_gripper_qvel"],
                }

                action_7dim = policy.get_action(input_state_dict)

                if policy.unreachable:
                    break

                obs, reward, done, info = env.step(action_7dim)

                if "agentview_image" in obs:
                    frames.append(prep_for_display(obs["agentview_image"], actual_instruction))

                if render_video:
                    try:
                        if "agentview_image" in obs:
                            cv2.imshow("Main Camera",
                                prep_for_display(obs["agentview_image"], actual_instruction))
                        if "robot0_eye_in_hand_image" in obs:
                            cv2.imshow("Gripper Camera",
                                prep_for_display(obs["robot0_eye_in_hand_image"], actual_instruction))
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                    except Exception:
                        pass

                # ── Check 3a: capture object baseline when gripper first opens
                if policy.state == 6 and placed_object_z is None:
                    placed_object_z = obs[f"{target_pick_label}_pos"][2]
                    placed_object_y = obs[f"{target_pick_label}_pos"][1]
                    debug_log_print("TrajectoryCheck",
                        f"State 6 entered — object Z={placed_object_z:.4f}  Y={placed_object_y:.4f}",
                        is_log_printed)

                # ── Check 3a (legacy): Z-rise while arm retracts ──────────────
                if (policy.state == 6
                        and placed_object_z is not None
                        and policy.open_steps <= 0):
                    z_rise = obs[f"{target_pick_label}_pos"][2] - placed_object_z
                    if z_rise > OBJECT_STUCK_Z_RISE:
                        object_stuck = True
                        debug_log_print("TrajectoryCheck",
                            f"CHECK 3 FAILED (Z-rise): object stuck — Z rose {z_rise:.4f}m",
                            is_log_printed)
                        break

                # ── Check 3b: post-done Y-shift extension window ──────────────
                # Runs AFTER done=True at state 6 to detect low-Z drops where
                # the object is dragged sideways when the gripper fingers open.
                if in_post_done:
                    post_done_counter += 1
                    if placed_object_y is not None:
                        y_shift = abs(obs[f"{target_pick_label}_pos"][1] - placed_object_y)
                        debug_log_print("TrajectoryCheck",
                            f"  extension step {post_done_counter}/{POST_DONE_EXTEND_STEPS} "
                            f"| ΔY={y_shift:.4f}",
                            is_log_printed)
                        if y_shift > OBJECT_STUCK_Y_SHIFT:
                            object_stuck = True
                            debug_log_print("TrajectoryCheck",
                                f"CHECK 3 FAILED (Y-shift): object dragged by gripper "
                                f"ΔY={y_shift:.4f}m at extension step {post_done_counter}",
                                is_log_printed)
                            break
                    if post_done_counter >= POST_DONE_EXTEND_STEPS:
                        break   # clean exit — Y was stable

                # ── Data collection (only for normal steps, not extension) ─────
                if step >= CAP_INDEX and not in_post_done:
                    all_actions.append(action_7dim)
                    all_gripper_states.append(obs["robot0_gripper_qpos"])
                    all_joint_states.append(obs["robot0_joint_pos"])
                    all_ee_states.append(np.hstack([
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                    ]))
                    all_states.append(env.sim.get_state().flatten())
                    all_robot_states.append(np.hstack([
                        obs["robot0_eef_pos"],
                        T.quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ]))
                    agentview_images.append(obs["agentview_image"])
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                    agentview_depths.append(obs["agentview_depth"])
                    eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])
                    all_rewards.append(reward)

                # ── done handling ─────────────────────────────────────────────
                if done:
                    success = (policy.state == 6) and (reward >= 1.0)
                    if success and not in_post_done:
                        # Enter Y-shift extension window instead of breaking
                        in_post_done = True
                        debug_log_print("TrajectoryCheck",
                            "done=True at state 6 — entering "
                            f"{POST_DONE_EXTEND_STEPS}-step Y-shift extension",
                            is_log_printed)
                    else:
                        # Failed episode, or extension already running (shouldn't
                        # normally re-trigger, but guard it anyway)
                        break

            env.close()
            cv2.destroyAllWindows()

            # ── evaluate all four checks ──────────────────────────────────────
            decisive_place = not policy.z_stalled    # Check 2
            object_placed  = not object_stuck         # Check 3 (both 3a and 3b)
            arm_reached    = not policy.unreachable   # Check 4

            fail_reasons = []
            if not success:        fail_reasons.append(f"task failed (state={policy.state})")
            if not decisive_place: fail_reasons.append("Z stalled in state 5")
            if not object_placed:  fail_reasons.append("object stuck to gripper")
            if not arm_reached:    fail_reasons.append(f"arm timed out in state {policy.state}")

            trajectory_ok = success and decisive_place and object_placed and arm_reached

            if trajectory_ok:
                cur_demo_data = {
                    "instruction":      actual_instruction,
                    "bddl_path":        result["bddl_path"],
                    "bddl_content":     result["bddl"],
                    "model_xml":        model_xml_initial,
                    "init_state":       init_state,
                    "seed":             seed,
                    "actions":          np.stack(all_actions),
                    "states":           np.stack(all_states),
                    "agentview_rgb":    np.stack(agentview_images),
                    "eye_in_hand_rgb":  np.stack(eye_in_hand_images),
                    "agentview_depth":  np.stack(agentview_depths),
                    "eye_in_hand_depth":np.stack(eye_in_hand_depths),
                    "gripper_states":   np.stack(all_gripper_states),
                    "joint_states":     np.stack(all_joint_states),
                    "ee_states":        np.stack(all_ee_states),
                    "robot_states":     np.stack(all_robot_states),
                }
                data_collected.append(cur_demo_data)
                break   # accept — stop retrying

            print(
                f"[RETRY] env={cur_env} attempt={attempt}/{MAX_RETRIES} "
                f"FAILED — {'; '.join(fail_reasons)}"
            )

        # ── log this episode (fires once, on the accepted attempt) ────────────
        if logger is not None:
            logger.log_episode(
                instruction=actual_instruction,
                object_type=actual_object_type,
                bowl_type=bow_type,
                seed=seed,
                success=success,
                final_state=policy.state,
                pick_pos=pick_pos,
                place_pos=place_pos,
                steps_taken=len(frames),
            )

        total_trajectory.append(frames)
        results.append(result)

    if is_save_video:
        save_video(total_trajectory, env_grid_len, actual_instruction, save_video_path)

    return total_trajectory, results, data_collected


# ──────────────────────────────────────────────────────────────────────────────
# Video saver
# ──────────────────────────────────────────────────────────────────────────────

def save_video(all_trajectories, env_grid_len, cur_instruction, save_folder_path):
    if save_folder_path is None:
        save_folder_path = "./"

    h, w = all_trajectories[0][0].shape[:2]
    grid_h = h * env_grid_len
    grid_w = w * env_grid_len

    title = cur_instruction.strip().replace(" ", "_")

    trajectory_len = max(len(t) for t in all_trajectories)
    total_env = len(all_trajectories)

    if trajectory_len > 1:
        output_file_path = os.path.join(save_folder_path, title + ".mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_file_path, fourcc, 30, (grid_w, grid_h))
    else:
        output_file_path = os.path.join(save_folder_path, title + ".png")

    for step_idx in tqdm(range(trajectory_len), desc="combining the frames"):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for env_idx in range(total_env):
            row = env_idx // env_grid_len
            col = env_idx  % env_grid_len
            traj = all_trajectories[env_idx]
            frame = traj[min(step_idx, len(traj) - 1)]
            grid[row*h:(row+1)*h, col*w:(col+1)*w] = frame

        if trajectory_len > 1:
            video_writer.write(grid)
        else:
            cv2.imwrite(output_file_path, grid)

    if trajectory_len > 1:
        video_writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    trajectory_output_path = "/Hyperplane/shuijie/trajectory_data_1"

    log_dir = os.path.join(trajectory_output_path, "logs")
    logger  = TrajectoryLogger(log_dir=log_dir)

    for testing_object_type in OBJECT_POOL:
        for testing_bowl_type in BOWL_POOL:

            cur_trajectory_output_path = os.path.join(
                trajectory_output_path, testing_object_type, testing_bowl_type
            )

            if os.path.exists(cur_trajectory_output_path):
                print(f"[DEBUG INFO trajectory generator]: {cur_trajectory_output_path} exists")
                continue

            os.makedirs(cur_trajectory_output_path, exist_ok=True)

            for testing_instruction in INSTRUCTION_TEMPLATES:
                policy = AutoGenPolicy(is_debugging=True)
                cur_total_trajectory, cur_results, cur_data_collected = generate_trajectory(
                    instruction=testing_instruction,
                    policy=policy,
                    object_type=testing_object_type,
                    bow_type=testing_bowl_type,
                    num_objects=10,
                    trajectory_len=300,
                    save_bddl=True,
                    render_video=False,
                    is_log_printed=True,
                    is_save_video=True,
                    save_video_path=cur_trajectory_output_path,
                    env_grid_len=4,
                    logger=logger,
                )

                logger.print_summary()
                logger.save()