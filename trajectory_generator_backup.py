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

Trajectory quality checks (all 3 must pass; failed attempts are retried):
    CHECK 1  reward >= 1.0 at episode end
    CHECK 2  state-5 Z reached target without stalling (gripper not pressing
             object into bowl).  Z-stall is detected when Z stops changing for
             Z_STALL_STEPS_MAX consecutive steps before reaching the target.
    CHECK 3  object did not stick to the gripper after opening — detected by
             watching for the object Z rising more than OBJECT_STUCK_Z_RISE
             once retraction begins.

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
    generate_random_rank_task_bddl, debug_log_print, get_obj_type
)
from random import randint
import numpy as np
import cv2
from tqdm import tqdm
from libero.libero.envs import OffScreenRenderEnv
from trajectory_logger import TrajectoryLogger

# ── gripper constants ──────────────────────────────────────────────────────────
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

# ── trajectory quality check constants ────────────────────────────────────────
# Check 2: Z-stall detection in state 5
Z_STALL_THRESHOLD   = 0.002    # metres per step — below this counts as stalled
Z_STALL_STEPS_MAX   = 8        # consecutive stalled steps before forcing 5→6

# Check 3: Object-stuck detection after gripper opens
OBJECT_STUCK_Z_RISE = 0.03     # metres — if object rises more than this, it's stuck

# Retry budget per (instruction, object_type, bowl_type) combination
MAX_RETRIES         = 20


# ──────────────────────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────────────────────

def prep_for_display(img, instruction=None, lineLen=30):
    bgr = img[..., ::-1]
    bgr = np.flipud(bgr).copy()
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
            cv2.putText(bgr, text, (5, 15 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bgr


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
            f"Need to implement _generate_action for {self.__class__.__name__}"
        )


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
    3  Lift Z to TRANSIT_CLEARANCE *while* sliding XY toward place target
       (diagonal arc avoids kinematic singularity on large-X retraction)
    4  Align XY precisely over place target at clearance height
    5  Lower Z to place height
    6  Open gripper, retract to neutral Z

    Input state dict keys
    ---------------------
    pick_pos             : (3,) absolute gripper target for grasping
    place_pos            : (3,) absolute gripper target for dropping
                           (already includes OBJECT_PLACE_OFFSET correction)
    object_height        : float  — z above table at which gripper closes
    cur_gripper_pos      : (3,) current end-effector position
    original_gripper_pos : (3,) rest position at episode start
    gripper_velocity     : (2,) joint velocities for the two gripper fingers

    Quality flag (read by the caller after the episode)
    ---------------------------------------------------
    z_stalled : bool
        Set True if state 5 was exited because Z stopped changing rather than
        because the target Z was reached.  Signals the gripper was pressing the
        object into the bowl → trajectory should be discarded (Check 2).
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

        # ── Check 2: Z-stall detection ─────────────────────────────────────
        self.z_stall_steps = 0
        self.prev_z        = None
        self.z_stalled     = False   # exposed to caller

    # ── helpers ───────────────────────────────────────────────────────────────

    def _transit_height(self, state_dict):
        """Absolute Z for the safe transit arc."""
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
        th   = self._transit_height(state_dict)

        # ── 0: XY aligned to pick target ──────────────────────────────────────
        if self.state == 0:
            if self._is_xy_reached(cur, pick):
                debug_log_print("AutoGenPolicy", "0→1: XY aligned to pick", self.is_debugging)
                self.state = 1
                return

        # ── 1: Z lowered to pick height ───────────────────────────────────────
        # Bug 1 fix: use PICK_Z_THRESHOLD (0.015) instead of the default 0.005.
        # The P-controller converges asymptotically; 0.005 caused 60+ crawling
        # steps because the small residual action was absorbed by joint damping.
        if self.state == 1:
            pick_z = TABLE_HEIGHT + GRIPPER_LENGTH + state_dict["object_height"]
            if self._is_z_reached(cur[2], pick_z, threshold=PICK_Z_THRESHOLD):
                debug_log_print("AutoGenPolicy", "1→2: at pick height", self.is_debugging)
                self.state = 2
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
                    return
                self.gripper_prev_vel = vel

        # ── 3: reached transit height → start XY alignment to place ──────────
        if self.state == 3:
            if self._is_z_reached(cur[2], th, threshold=0.02):
                debug_log_print("AutoGenPolicy", "3→4: at transit height, moving to place XY", self.is_debugging)
                self.state = 4
                return

        # ── 4: XY aligned to place target ─────────────────────────────────────
        if self.state == 4:
            if self._is_xy_reached(cur, plc):
                debug_log_print("AutoGenPolicy", "4→5: XY aligned to place target", self.is_debugging)
                self.state = 5
                # Reset Z-stall tracker fresh on entering state 5
                self.z_stall_steps = 0
                self.prev_z        = None
                return

        # ── 5: Z lowered to place height ──────────────────────────────────────
        # CHECK 2: Z-stall guard — if Z freezes before reaching the target the
        # gripper is pressing the object into the bowl.  Force 5→6 and flag it.
        if self.state == 5:
            if self.prev_z is not None:
                z_delta = abs(cur[2] - self.prev_z)
                if z_delta < Z_STALL_THRESHOLD:
                    self.z_stall_steps += 1
                else:
                    self.z_stall_steps = 0   # reset on any meaningful movement

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
                # if self._is_xy_reached(cur, plc):
                    debug_log_print("AutoGenPolicy", "5→6: at place height, opening gripper", self.is_debugging)
                    self.state = 6
            return

        # ── 6: gripper open, done ─────────────────────────────────────────────
        # (terminal state — no further transitions needed)

    # ── action generation ─────────────────────────────────────────────────────

    def _generate_action(self, state_dict):
        self._update_state(state_dict)

        cur  = state_dict["cur_gripper_pos"]
        pick = state_dict["pick_pos"]
        plc  = state_dict["place_pos"]
        oh   = state_dict["object_height"]
        th   = self._transit_height(state_dict)

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

        # ── 3: DIAGONAL ARC — lift Z while sliding XY toward place ────────────
        elif self.state == 3:
            action[2]  = th - cur[2]
            # action[:2] = plc[:2] - cur[:2]
            action[:2] = 0.0
            action[6]  = GRIPPER_CLOSE
            debug_log_print("AutoGenPolicy",
                f"state=3 | lifting+sliding | Δz={action[2]:.4f} | target_th={th:.4f}", self.is_debugging)

        # ── 4: align XY over place target at transit height ──────────────────
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
                action[2]  = th - cur[2]
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


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_trajectory(
    instruction,
    policy: Policy,
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
    logger: TrajectoryLogger = None,   # ← unchanged
):
    total_env = env_grid_len * env_grid_len
    total_trajectory = []
    results = []

    for cur_env in range(total_env):

        # ── retry loop: keep trying until all 3 checks pass or budget runs out
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

            env = OffScreenRenderEnv(
                bddl_file_name=result["bddl_path"],
                robots=["Panda"],
                camera_heights=256,
                camera_widths=256,
            )

            seed = randint(0, 1000)
            env.seed(seed)
            obs = env.reset()

            params    = OBJECT_PICK_PARAMS[actual_object_type]
            pick_pos  = obs[f"{target_pick_label}_pos"] + np.array([params[0], params[1], 0.0])
            place_pos = obs[f"{target_place_label}_pos"] + BOWL_PLACE_POS[bow_type] + OBJECT_PLACE_OFFSET[actual_object_type]

            original_gripper_pos = obs["robot0_eef_pos"]
            frames  = []
            success = False

            # ── Check 3 tracking ──────────────────────────────────────────────
            # Record the object Z the moment state 6 is first entered, then
            # watch for it rising once the arm retracts.
            placed_object_z = None
            object_stuck    = False

            for step in range(trajectory_len):
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

                # ── Check 3a: capture object Z baseline when gripper first opens
                if policy.state == 6 and placed_object_z is None:
                    placed_object_z = obs[f"{target_pick_label}_pos"][2]
                    debug_log_print("TrajectoryCheck",
                        f"State 6 entered — object Z baseline: {placed_object_z:.4f}",
                        is_log_printed)

                # ── Check 3b: once arm retracts, watch for object rising ───────
                # policy.open_steps counts down each step in state 6;
                # <= 0 means retraction has started.
                if (policy.state == 6
                        and placed_object_z is not None
                        and policy.open_steps <= 0):
                    z_rise = obs[f"{target_pick_label}_pos"][2] - placed_object_z
                    if z_rise > OBJECT_STUCK_Z_RISE:
                        object_stuck = True
                        debug_log_print("TrajectoryCheck",
                            f"CHECK 3 FAILED: object stuck — Z rose {z_rise:.4f}m",
                            is_log_printed)
                        break

                if done:
                    success = (reward >= 1.0)   # LIBERO convention: reward=1 on success
                    # break

            env.close()
            cv2.destroyAllWindows()

            # ── evaluate all three checks ─────────────────────────────────────
            decisive_place = not policy.z_stalled   # Check 2
            object_placed  = not object_stuck        # Check 3
            trajectory_ok  = success and decisive_place and object_placed

            if trajectory_ok:
                break   # accept — stop retrying

            fail_reasons = []
            if not success:        fail_reasons.append(f"task failed (state={policy.state})")
            if not decisive_place: fail_reasons.append("Z stalled in state 5")
            if not object_placed:  fail_reasons.append("object stuck to gripper")
            print(
                f"[RETRY] env={cur_env} attempt={attempt}/{MAX_RETRIES} "
                f"FAILED — {'; '.join(fail_reasons)}"
            )

        # ── log this episode (unchanged — fires once, on the accepted attempt) ─
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

    return total_trajectory, results


# In save_video — clamp to shortest trajectory, pad short ones with last frame
def save_video(all_trajectories, env_grid_len, cur_instruction, save_folder_path):
    if save_folder_path is None:
        save_folder_path = "./"

    h, w = all_trajectories[0][0].shape[:2]
    grid_h = h * env_grid_len
    grid_w = w * env_grid_len

    title = cur_instruction.strip().replace(" ", "_")

    # Use the LONGEST trajectory, repeating last frame for episodes that ended early
    trajectory_len = max(len(t) for t in all_trajectories)

    total_env = len(all_trajectories)

    if trajectory_len > 1:
        output_file_path = os.path.join(save_folder_path, title + "avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(output_file_path, fourcc, 30, (grid_w, grid_h))
    else:
        output_file_path = os.path.join(save_folder_path, title + ".png")

    for step_idx in tqdm(range(trajectory_len), desc="combining the frames"):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for env_idx in range(total_env):
            row = env_idx // env_grid_len
            col = env_idx  % env_grid_len
            traj = all_trajectories[env_idx]
            # Clamp: repeat last frame if this episode ended early
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

    INSTRUCTION_TEMPLATES = [
        # egocentric pick tasks: many same object + 1 bowl (16 tasks)
        "Pick the closest object and place in the bowl.",
        "Pick the furtherest object and place in the bowl.",
        # "Pick the 1st closest object and place in the bowl.",
        # "Pick the 2nd closest object and place in the bowl.",
        # "Pick the 3rd closest object and place in the bowl.",
        # "Pick the 4th closest object and place in the bowl.",
        # "Pick the 5th closest object and place in the bowl.",
        # "Pick the 6th closest object and place in the bowl.",
        # "Pick the 7th closest object and place in the bowl.",
        # "Pick the 1st furtherest object and place in the bowl.",
        # "Pick the 2nd furtherest object and place in the bowl.",
        # "Pick the 3rd furtherest object and place in the bowl.",
        # "Pick the 4th furtherest object and place in the bowl.",
        # "Pick the 5th furtherest object and place in the bowl.",
        # "Pick the 6th furtherest object and place in the bowl.",
        # "Pick the 7th furtherest object and place in the bowl.",

        # egocentric place tasks: 1 object + many bowls (16 tasks)
        "Pick the object and place in the closest bowl.",
        "Pick the object and place in the furtherest bowl.",
        # "Pick the object and place in the 1st closest bowl.",
        # "Pick the object and place in the 2nd closest bowl.",
        # "Pick the object and place in the 3rd closest bowl.",
        # "Pick the object and place in the 4th closest bowl.",
        # "Pick the object and place in the 5th closest bowl.",
        # "Pick the object and place in the 6th closest bowl.",
        # "Pick the object and place in the 7th closest bowl.",
        # "Pick the object and place in the 1st furtherest bowl.",
        # "Pick the object and place in the 2nd furtherest bowl.",
        # "Pick the object and place in the 3rd furtherest bowl.",
        # "Pick the object and place in the 4th furtherest bowl.",
        # "Pick the object and place in the 5th furtherest bowl.",
        # "Pick the object and place in the 6th furtherest bowl.",
        # "Pick the object and place in the 7th furtherest bowl.",

        # allocentric pick: many same object + 1 bowl (16 tasks)
        "Pick the object closest to the bowl and place in the bowl.",
        "Pick the object furtherest to the bowl and place in the bowl.",
        # "Pick the 1st object closest to the bowl and place in the bowl.",
        # "Pick the 2nd object closest to the bowl and place in the bowl.",
        # "Pick the 3rd object closest to the bowl and place in the bowl.",
        # "Pick the 4th object closest to the bowl and place in the bowl.",
        # "Pick the 5th object closest to the bowl and place in the bowl.",
        # "Pick the 6th object closest to the bowl and place in the bowl.",
        # "Pick the 7th object closest to the bowl and place in the bowl.",
        # "Pick the 1st object furtherest to the bowl and place in the bowl.",
        # "Pick the 2nd object furtherest to the bowl and place in the bowl.",
        # "Pick the 3rd object furtherest to the bowl and place in the bowl.",
        # "Pick the 4th object furtherest to the bowl and place in the bowl.",
        # "Pick the 5th object furtherest to the bowl and place in the bowl.",
        # "Pick the 6th object furtherest to the bowl and place in the bowl.",
        # "Pick the 7th object furtherest to the bowl and place in the bowl.",

        # allocentric place: 1 object + many bowls (16 tasks)
        "Pick the object and place in the bowl closest to it.",
        "Pick the object and place in the bowl furtherest from it.",
        # "Pick the object and place in the 1st bowl closest to it.",
        # "Pick the object and place in the 2nd bowl closest to it.",
        # "Pick the object and place in the 3rd bowl closest to it.",
        # "Pick the object and place in the 4th bowl closest to it.",
        # "Pick the object and place in the 5th bowl closest to it.",
        # "Pick the object and place in the 6th bowl closest to it.",
        # "Pick the object and place in the 7th bowl closest to it.",
        # "Pick the object and place in the 1st bowl furtherest from it.",
        # "Pick the object and place in the 2nd bowl furtherest from it.",
        # "Pick the object and place in the 3rd bowl furtherest from it.",
        # "Pick the object and place in the 4th bowl furtherest from it.",
        # "Pick the object and place in the 5th bowl furtherest from it.",
        # "Pick the object and place in the 6th bowl furtherest from it.",
        # "Pick the object and place in the 7th bowl furtherest from it.",

        # pick by feature: many different objects + 1 bowl (12 tasks)
        "Pick the largest object and place in the bowl.",
        "Pick the smallest object and place in the bowl.",
        # "Pick the 1st largest object and place in the bowl.",
        # "Pick the 2nd largest object and place in the bowl.",
        # "Pick the 3rd largest object and place in the bowl.",
        # "Pick the 4th largest object and place in the bowl.",
        # "Pick the 5th largest object and place in the bowl.",
        # "Pick the 1st smallest object and place in the bowl.",
        # "Pick the 2nd smallest object and place in the bowl.",
        # "Pick the 3rd smallest object and place in the bowl.",
        # "Pick the 4th smallest object and place in the bowl.",
        # "Pick the 5th smallest object and place in the bowl.",

        # place by feature: 1 object + different bowls (8 tasks)
        "Pick the object and place in the largest bowl.",
        "Pick the object and place in the smallest bowl.",
        # "Pick the object and place in the 1st largest bowl.",
        # "Pick the object and place in the 2nd largest bowl.",
        # "Pick the object and place in the 3rd largest bowl.",
        # "Pick the object and place in the 1st smallest bowl.",
        # "Pick the object and place in the 2nd smallest bowl.",
        # "Pick the object and place in the 3rd smallest bowl.",

        # middle pick/place (2 tasks)
        "Pick the object in the middle and place in the bowl.",
        "Pick the object and place in the bowl in the middle.",
    ]

    TABLE_HEIGHT = 0.9

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
    #   For centred grasps this is [0, 0, 0].
    #   For the ramekin the gripper contacts the rim at +y, so the object hangs
    #   -y relative to the gripper centre; negate the pick y-offset here to
    #   correct the landing position.
    #   Tune z_offset per-object if the drop height needs adjustment.

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
        "akita_black_bowl": np.array([0.0, 0.0, 0.0]),
        "plate":            np.array([0.0, 0.0, 0.0]),
    }

    OBJECT_POOL = [
        "milk",
        "orange_juice",
        "ketchup",
        "alphabet_soup",
        "moka_pot",
        "tomato_sauce",
        "glazed_rim_porcelain_ramekin",
        "butter",
    ]

    BOWL_POOL = [
        "akita_black_bowl",
        "white_bowl",
        "plate"
    ]

    trajectory_output_path = (
        "/Hyperplane/shuijie/trajectory_data_1"
    )

    log_dir = os.path.join(trajectory_output_path, "logs")
    logger  = TrajectoryLogger(log_dir=log_dir)

    for testing_object_type in OBJECT_POOL:
        if os.path.exists(os.path.join(trajectory_output_path, testing_object_type)):
            print(f"[DEBUG INFO trajectory generator]: path to {testing_object_type} exists")
            continue

        for testing_bowl_type in BOWL_POOL:

            cur_trajectory_output_path = os.path.join(trajectory_output_path, testing_object_type, testing_bowl_type)

            if os.path.exists(cur_trajectory_output_path):
                print(f"[DEBUG INFO trajectory generator]: {cur_trajectory_output_path} exists")
                continue

            os.makedirs(cur_trajectory_output_path, exist_ok=True)

            for testing_instruction in INSTRUCTION_TEMPLATES:
                policy = AutoGenPolicy(is_debugging=True)
                generate_trajectory(
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
                    logger=logger,          # ← pass it in
                )

                logger.print_summary()
                logger.save()