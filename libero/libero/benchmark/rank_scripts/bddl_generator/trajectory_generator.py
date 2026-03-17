"""
Since all tasks in LIBERO_RANK are simple pick and place, in theory we could generate the correct trajectory to
complete all the tasks if we have the location of the objects.

1. we need a naive trajectory generator that does the following.
    1. move the end effector to the top of the target object
    2. grab the object firmly and move up 
    3. move to the target bowl instance directly above
    4. lower to appropriate height before opening the gripper.

2. For each object, the wa its placed its always the same. Hence the idea is that given an object, the target gripping point should remain consistent.
    1. for each object, design a fixed gripping point (assume the object global pos as the centre).
    2. for each bowl, design a fixed dropping point (assume the bowl global pos as the centre).

Stillneed to complete to trajectory

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
libero_path =  os.path.join(os.path.dirname(__file__), "..","..", "..", "..", "..")
print(f"libero path: {libero_path}")

import sys
sys.path.insert(0, libero_path)


from libero.libero.benchmark.rank_scripts.bddl_generator import generate_random_rank_task_bddl, debug_log_print, get_obj_type
from random import randint
import numpy as np


import cv2
import numpy as np
from tqdm import tqdm
from random import randint
from libero.libero.envs import OffScreenRenderEnv

GRIPPER_OPEN   =  -1.0
GRIPPER_CLOSE  = 1.0
GRIP_HOLD_STEPS = 10   # steps to dwell while opening/closing
GRIPPER_LENGTH = 0.0


def prep_for_display(img, instruction=None, lineLen=30):
    bgr = img[..., ::-1]
    bgr = np.flipud(bgr).copy()
    if instruction is not None:
        words = instruction.split()
        lines, line = [], []
        for word in words:
            line.append(word)
            if len(" ".join(line)) > lineLen:  # adjust threshold to taste
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


class Policy():
    def __init__(self, input_state_dict_keys: list, is_debugging: bool = False):
        self.input_state_dict_keys = input_state_dict_keys
        self.is_debugging = is_debugging

    """
    Check if input state contains all the necessary keys to generate action
    """
    def check_input_state_dict_keys(self, state_dict: dict):
        return  all(required_state_keys in state_dict for required_state_keys in self.input_state_dict_keys)
    
    def _generate_action(self, state_dict):
        raise NotImplementedError(f"Need to implement get action function for class {self.__class__.__name__}")
        
    def get_action(self, state_dict):
        if not self.check_input_state_dict_keys(state_dict):
            absent_keys = [required_state_keys not in state_dict for required_state_keys in self.input_state_dict_keys]
            raise ValueError(f"Polcy class class {self.__class__.__name__} demands following keys {absent_keys} to be present in input state_dict to generate action.")

        # generate actions if state_dict contains all the required keys
        return self._generate_action(state_dict)


class AutoGenPolicy(Policy):
    def __init__(self, is_debugging: bool = False):
        state_dict_keys = ["pick_pos", "place_pos", "object_height","cur_gripper_pos", "original_gripper_pos"]
        super().__init__(state_dict_keys, is_debugging)
        self.state = 0
        self.grip_hold_counter = 0
        self.reach_above_object = False
        self.gripper_prev_vel = 0.0

    # considered reached if each axis difference less than threshold
    def _is_pos_reached(self, cur_pos, target_pos, threshold=0.02):
        return all(abs(diff) <= threshold for diff in cur_pos - target_pos)


    """
    update state first before getting action
    """
    def _update_state(self, state_dict):
        if self.state == 0:  # ── move to pick position, gripper open
            if self._is_pos_reached(state_dict["cur_gripper_pos"][:2], state_dict["pick_pos"][:2]):
                debug_log_print(
                    function_name="AutoGenPolicy._update_state",
                    debug_message=f"state 0 -> 1: we have reach the top of the object",
                    is_debugging=self.is_debugging
                )
                self.state += 1
                return
        
        if self.state == 1:  # ── lowered gripper to object
            pick_height = TABLE_HEIGHT + GRIPPER_LENGTH + state_dict["object_height"]
            if state_dict["cur_gripper_pos"][2] - pick_height <= 0.0:
                debug_log_print(
                    function_name="AutoGenPolicy._update_state",
                    debug_message=f"state 1 -> 2: we have reach the picking point of the object",
                    is_debugging=self.is_debugging
                )
                self.state += 1
                return
        
        if self.state == 2:  # ── gripping object, stop if velocity decreases
            if state_dict["gripper_velocity"][1] <  self.gripper_prev_vel:
                debug_log_print(
                    function_name="AutoGenPolicy._update_state",
                    debug_message=f"state 2 -> 3: grepper slowed down, we have grabed the object",
                    is_debugging=self.is_debugging
                )
                self.state += 1
                return
            
            # instead update gripper recorded velocity
            self.gripper_prev_vel = state_dict["gripper_velocity"][1]

        if self.state == 3:  # ── gripping object, stop if velocity decreases
            lift_height = TABLE_HEIGHT + GRIPPER_LENGTH + state_dict["object_height"] * 2
            if state_dict["cur_gripper_pos"][2] - lift_height <= 0.0:
                debug_log_print(
                    function_name="AutoGenPolicy._update_state",
                    debug_message=f"state 3 -> 4: we can move towards the plate",
                    is_debugging=self.is_debugging
                )
                self.state += 1
                return
            
        if self.state == 4:  # ── move to pick position, gripper open
            if self._is_pos_reached(state_dict["cur_gripper_pos"][:2], state_dict["place_pos"][:2]):
                debug_log_print(
                    function_name="AutoGenPolicy._update_state",
                    debug_message=f"state 3 -> 4: we have reach the top of the bowl",
                    is_debugging=self.is_debugging
                )
                self.state += 1
                return

    """
    State dict contains various data for the policy agent to generate action.
    For auto policy this is needed
    """
    def _generate_action(self, state_dict):
        # first update the state if needed
        self._update_state(state_dict)
        action_7dim = np.zeros(7)

        if self.state == 0:  # ── move to pick position, gripper open
            action_7dim[:2] = state_dict["pick_pos"][:2] - state_dict["cur_gripper_pos"][:2]

            # keep the arm z-axis above the objects on the table, if value is negative, move up, else stay zero
            action_7dim[2] = -min(0.0, state_dict["cur_gripper_pos"][2] - state_dict["object_height"] - TABLE_HEIGHT)
            action_7dim[6]  = GRIPPER_OPEN
            debug_log_print(
                function_name="AutoGenPolicy._generate_action",
                debug_message=f"state=0 | aligning XY to pick | delta_xyz={action_7dim[:3]} | target_xy={state_dict['pick_pos'][:2]}",
                is_debugging=self.is_debugging
            )
        
        if self.state == 1:  # ── lower the gripper to get the object
            # assume the height to pick the object is halfway the height
            # need to consider the length of the gripper
            pick_height = TABLE_HEIGHT + GRIPPER_LENGTH + state_dict["object_height"]
            action_7dim[2] = pick_height - state_dict["cur_gripper_pos"][2]

            # still need to make sure the gripper stays above the object
            action_7dim[:2] = state_dict["pick_pos"][:2] - state_dict["cur_gripper_pos"][:2]

            debug_log_print(
                function_name="AutoGenPolicy._generate_action",
                debug_message=f"state=1 | lowering Z to pick | delt_z={action_7dim[2]} | target_z={pick_height}",
                is_debugging=self.is_debugging
            )

        if self.state == 2: # ── close the gripper untill cannot close
            action_7dim[6] = GRIPPER_CLOSE
            gripper_vel = state_dict["gripper_velocity"]
            debug_log_print(
                function_name="AutoGenPolicy._generate_action",
                debug_message=f"state=2 | closing gripper | gripper_vel={gripper_vel}",
                is_debugging=self.is_debugging
            )

        if self.state == 3: # ── lift the object up, must be above all other objects to prevent collision
            
            lift_height = TABLE_HEIGHT + GRIPPER_LENGTH + state_dict["object_height"] * 2

            # lift_height = 1.2

            # lift the gripper up with no change in xy location
            action_7dim[2] = lift_height - state_dict["cur_gripper_pos"][2]
            action_7dim[:2] = state_dict["pick_pos"][:2] - state_dict["cur_gripper_pos"][:2]
            
            debug_log_print(
                function_name="AutoGenPolicy._generate_action",
                debug_message=f"state=3 | lifting gripper up to prevent collision |  delt_z={action_7dim[2]} | target_z={lift_height}",
                is_debugging=self.is_debugging
            )
        
        if self.state == 4:
            action_7dim[:2] = state_dict["place_pos"][:2] - state_dict["cur_gripper_pos"][:2]

            # maintain height
            lift_height = TABLE_HEIGHT + GRIPPER_LENGTH + state_dict["object_height"] * 2
            action_7dim[2] =  -min(0.0, state_dict["cur_gripper_pos"][2] - lift_height)

            debug_log_print(
                function_name="AutoGenPolicy._generate_action",
                debug_message=f"state=4 | aligning XY to place | delta_xyz={action_7dim[:3]} | target_xy={state_dict['place_pos'][:2]}",
                is_debugging=self.is_debugging
            )

        if self.state == 5:
            action_7dim[6] = GRIPPER_OPEN
            gripper_vel = state_dict["gripper_velocity"]
            debug_log_print(
                function_name="AutoGenPolicy._generate_action",
                debug_message=f"state=5 | opening gripper | gripper_vel={gripper_vel}",
                is_debugging=self.is_debugging
            )
        
        action_7dim[:3] = np.clip(action_7dim[:3] * 10, -1, 1)

        return action_7dim
        

def generate_trajectory(instruction, policy:Policy, object_type=None, bow_type=None,num_objects=10,
                         trajectory_len=100, save_bddl=True, output_path=None, render_video=True, is_log_printed=False):
    
    result = generate_random_rank_task_bddl(language=instruction, object_type=object_type, bow_type=bow_type,
                                             num_objects=num_objects, save_bddl=True, output_path=output_path, is_debugging=is_log_printed)

    actual_instruction = result["resolved_language"]

    # label contains both the object type and the index
    target_pick_label = result["target_object"]  # e.g. "cookies_1"
    target_place_label = result["target_place"]

    pick_obj_type = get_obj_type(target_pick_label)
    place_bowl_type = get_obj_type(target_place_label)
    

    env = OffScreenRenderEnv(bddl_file_name=result["bddl_path"], robots=["Panda"],
                            camera_heights=256, camera_widths=256)
    
    seed = randint(0, 1000)
    env.seed(seed)
    obs = env.reset()

    # this are the 2 locations that the gripper must reach

    debug_log_print(function_name="generate_trajectory", debug_message=f"observation contains {obs.keys()}", is_debugging=is_log_printed)
    
    pick_pos = obs[f"{target_pick_label}_pos"] + OBJECT_PICK_POS[object_type]
    place_pos = obs[f"{target_place_label}_pos"]

    debug_log_print(function_name="generate_trajectory", debug_message=f"pick pos {pick_pos} and place pos {place_pos}", is_debugging=is_log_printed)

    

    # considered reached if each axis difference less than threshold
    def is_pos_reached(cur_pos, target_pos, threshold=0.02):
        return all(abs(diff) <= threshold for diff in cur_pos - target_pos)



    # start with the pick pos
    # State machine
    original_gripper_pos = obs["robot0_eef_pos"] # for end trajectory
    test = obs["robot0_gripper_qpos"]


    frames = []
    # for step in tqdm(range(trajectory_len), desc="Moving to target"):
    for step in range(trajectory_len):
        cur_gripper_pos = obs["robot0_eef_pos"]
        
        # debug_log_print(function_name="generate_trajectory", debug_message=f"cur_gripper_pos {cur_gripper_pos}", is_debugging=is_log_printed)
        # debug_log_print(function_name="generate_trajectory", debug_message=f"testing robot0_gripper_qpos {test}", is_debugging=is_log_printed)


        input_state_dict = {
            "pick_pos": pick_pos,
            "place_pos": place_pos,
            "cur_gripper_pos": cur_gripper_pos,
            "original_gripper_pos": original_gripper_pos,
            "object_height": OBJECT_HEIGHT[object_type],
            "gripper_velocity":  obs["robot0_gripper_qvel"]
        }

        action_7dim = policy.get_action(input_state_dict)
    
        obs, reward, done, info = env.step(action_7dim)

        if "agentview_image" in obs:
            frames.append(prep_for_display(obs["agentview_image"], actual_instruction))

        if render_video:
            try:
                if "agentview_image" in obs:
                    cv2.imshow("Main Camera", prep_for_display(obs["agentview_image"], actual_instruction))
                if "robot0_eye_in_hand_image" in obs:
                    cv2.imshow("Gripper Camera", prep_for_display(obs["robot0_eye_in_hand_image"], actual_instruction))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception:
                pass

        if done:
            break

    env.close()
    cv2.destroyAllWindows()
    return frames, result


if __name__ =="__main__":
    # 81 tasks in total
    INSTRUCTION_TEMPLATES = [
        # egocentric pick tasks: many same object + 1 bowl (16 tasks)
        "Pick the closest object and place in the bowl.",
        "Pick the furtherest object and place in the bowl.",
        "Pick the 1st closest object and place in the bowl.",
        "Pick the 2nd closest object and place in the bowl.",
        "Pick the 3rd closest object and place in the bowl.",
        "Pick the 4th closest object and place in the bowl.",
        "Pick the 5th closest object and place in the bowl.",
        "Pick the 6th closest object and place in the bowl.",
        "Pick the 7th closest object and place in the bowl.",
        "Pick the 1st furtherest object and place in the bowl.",
        "Pick the 2nd furtherest object and place in the bowl.",
        "Pick the 3rd furtherest object and place in the bowl.",
        "Pick the 4th furtherest object and place in the bowl.",
        "Pick the 5th furtherest object and place in the bowl.",
        "Pick the 6th furtherest object and place in the bowl.",
        "Pick the 7th furtherest object and place in the bowl.",
        # egocentric place tasks: 1 object + many bowls (16 tasks)
        "Pick the object and place in the closest bowl.",
        "Pick the object and place in the furtherest bowl.",
        "Pick the object and place in the 1st closest bowl.",
        "Pick the object and place in the 2nd closest bowl.",
        "Pick the object and place in the 3rd closest bowl.",
        "Pick the object and place in the 4th closest bowl.",
        "Pick the object and place in the 5th closest bowl.",
        "Pick the object and place in the 6th closest bowl.",
        "Pick the object and place in the 7th closest bowl.",
        "Pick the object and place in the 1st furtherest bowl.",
        "Pick the object and place in the 2nd furtherest bowl.",
        "Pick the object and place in the 3rd furtherest bowl.",
        "Pick the object and place in the 4th furtherest bowl.",
        "Pick the object and place in the 5th furtherest bowl.",
        "Pick the object and place in the 6th furtherest bowl.",
        "Pick the object and place in the 7th furtherest bowl.",
        # allocentric pick: many same object + 1 bowl (16 tasks)
        "Pick the object closest to the bowl and place in the bowl.",
        "Pick the object furtherest to the bowl and place in the bowl.",
        "Pick the 1st object closest to the bowl and place in the bowl.",
        "Pick the 2nd object closest to the bowl and place in the bowl.",
        "Pick the 3rd object closest to the bowl and place in the bowl.",
        "Pick the 4th object closest to the bowl and place in the bowl.",
        "Pick the 5th object closest to the bowl and place in the bowl.",
        "Pick the 6th object closest to the bowl and place in the bowl.",
        "Pick the 7th object closest to the bowl and place in the bowl.",
        "Pick the 1st object furtherest to the bowl and place in the bowl.",
        "Pick the 2nd object furtherest to the bowl and place in the bowl.",
        "Pick the 3rd object furtherest to the bowl and place in the bowl.",
        "Pick the 4th object furtherest to the bowl and place in the bowl.",
        "Pick the 5th object furtherest to the bowl and place in the bowl.",
        "Pick the 6th object furtherest to the bowl and place in the bowl.",
        "Pick the 7th object furtherest to the bowl and place in the bowl.",
        # allocentric place: 1 object + many bowls (16 tasks)
        "Pick the object and place in the bowl closest to it.",
        "Pick the object and place in the bowl furtherest from it.",
        "Pick the object and place in the 1st bowl closest to it.",
        "Pick the object and place in the 2nd bowl closest to it.",
        "Pick the object and place in the 3rd bowl closest to it.",
        "Pick the object and place in the 4th bowl closest to it.",
        "Pick the object and place in the 5th bowl closest to it.",
        "Pick the object and place in the 6th bowl closest to it.",
        "Pick the object and place in the 7th bowl closest to it.",
        "Pick the object and place in the 1st bowl furtherest from it.",
        "Pick the object and place in the 2nd bowl furtherest from it.",
        "Pick the object and place in the 3rd bowl furtherest from it.",
        "Pick the object and place in the 4th bowl furtherest from it.",
        "Pick the object and place in the 5th bowl furtherest from it.",
        "Pick the object and place in the 6th bowl furtherest from it.",
        "Pick the object and place in the 7th bowl furtherest from it.",
        # pick by feature: many different object + 1 bowl (12 tasks)
        "Pick the largest object and place in the bowl.",
        "Pick the smallest object and place in the bowl.",
        "Pick the 1st largest object and place in the bowl.",
        "Pick the 2nd largest object and place in the bowl.",
        "Pick the 3rd largest object and place in the bowl.",
        "Pick the 4th largest object and place in the bowl.",
        "Pick the 5th largest object and place in the bowl.",
        "Pick the 1st smallest object and place in the bowl.",
        "Pick the 2nd smallest object and place in the bowl.",
        "Pick the 3rd smallest object and place in the bowl.",
        "Pick the 4th smallest object and place in the bowl.",
        "Pick the 5th smallest object and place in the bowl.",
        # place by feature: 1 object + different bowl (8 tasks)
        "Pick the object and place in the largest bowl.",
        "Pick the object and place in the smallest bowl.",
        "Pick the object and place in the 1st largest bowl.",
        "Pick the object and place in the 2nd largest bowl.",
        "Pick the object and place in the 3rd largest bowl.",
        "Pick the object and place in the 1st smallest bowl.",
        "Pick the object and place in the 2nd smallest bowl.",
        "Pick the object and place in the 3rd smallest bowl.",
        # middle pick: 3 different object with col restriction + one bowl (2 tasks)
        "Pick the object in the middle and place in the bowl.",
        "Pick the object and place in the bowl in the middle."
    ]


    TABLE_HEIGHT = 0.9
    # add to object actual sim position to get the sim pick position
    OBJECT_PICK_POS = {
        "milk": [0.0, 0.0, 0.01],
        "moka_pot": [0.0, 0.0, 0.0],
        "glazed_rim_porcelain_ramekin": [0.0, 0.0, 0.0],
        "tomato_sauce": [0.0, 0.0, 0.0],
        "alphabet_soup": [0.0, 0.0, 0.0],
        "butter": [0.0, 0.0, 0.0],
        "ketchup": [0.0, 0.0, 0.0],
        "orange_juice": [0.0, 0.0, 0.0],
    }


    OBJECT_HEIGHT= {
        "milk": 0.1, # 1.04 barely touch the top, so its should be 0.07. pick up at 1.00, need to raise it twice the height (1.11)
        "moka_pot": 0,
        "glazed_rim_porcelain_ramekin": 0,
        "tomato_sauce": 0,
        "alphabet_soup": 0,
        "butter": 0.02,
        "ketchup": 0,
        "orange_juice": 0,
    }

    # add to bowl actual sim position to get the sim place position
    BOWL_PLACE_POS = {
        "white_bowl": [0.0, 0.0, 0.03],
        "akita_black_bowl": [0.0, 0.0, 0.0],
        "plate": [0.0, 0.0, 0.0],
    }

    OBJECT_POOL = [
        # "milk",
        # "moka_pot",
        # "glazed_rim_porcelain_ramekin",
        # "tomato_sauce",
        # "alphabet_soup",
        "butter",
        # "ketchup",
        # "orange_juice",
    ]

    BOWL_POOL = [
        "white_bowl",
        # "akita_black_bowl",
        # "plate",
    ]

    trajectory_output_path = "E:/Shui Jie/PHD_school/research/code/global_depth_order_benchmark/LIBERO-DEPTH-ORDER/trajectory_data"

    
    for testing_instruction in INSTRUCTION_TEMPLATES:

        for testing_object_type in OBJECT_POOL:
            for testing_bowl_type in BOWL_POOL:
                policy = AutoGenPolicy(is_debugging=True)

                generate_trajectory(instruction=testing_instruction, policy=policy, object_type=testing_object_type, bow_type=testing_bowl_type,
                                    num_objects=10, trajectory_len=1000, save_bddl=True, output_path=None, render_video=True, is_log_printed=True)





    # # this is the actual sim position (x, y, z) position
    # target_object_type = target_pick_label.split('_')[0]
    # pick_pos  = obs[f"{target_pick_label}_pos"] 


#     target_key = result["target_object"]  # e.g. "cookies_1"

#     target_pos  = obs[f"{target_key}_pos"]  # actual sim position (x, y, z)

#     frames = []

#     for step in tqdm(range(trajectory_len), desc="Moving to target"):
#         robot_eef_pos = obs["robot0_eef_pos"]

#         # Use actual target_pos from obs — update each step in case of physics drift
#         target_pos = obs[f"{target_key}_pos"]
#         delta_pos  = target_pos - robot_eef_pos# keep height constant

#         # print(delta_pos)

#         if sum(delta_pos[:2]) < 0.03:
#             # move to the target bowl after reaching the target pick
#             target_key = result["target_place"]

#         action_7dim = np.zeros(7)
#         action_7dim[:3] = np.clip(delta_pos * 10, -1, 1)
#         action_7dim[2] = 0.0

#         obs, reward, done, info = env.step(action_7dim)

#         if "agentview_image" in obs:
#             frames.append(prep_for_display(obs["agentview_image"], actual_instruction))

#         if render_video:
#             try:
#                 if "agentview_image" in obs:
#                     cv2.imshow("Main Camera", prep_for_display(obs["agentview_image"], actual_instruction))
#                 if "robot0_eye_in_hand_image" in obs:
#                     cv2.imshow("Gripper Camera", prep_for_display(obs["robot0_eye_in_hand_image"], actual_instruction))
#                 if cv2.waitKey(1) & 0xFF == 27:
#                     break
#             except Exception:
#                 pass

#     env.close()
#     cv2.destroyAllWindows()

# if save_video:

#     if save_folder_path is None:
#         save_folder_path = "./"


#     # combine frames to grid
#     h, w = all_trajectories[0][0].shape[:2]
#     grid_h = h * env_grid_len
#     grid_w = w * env_grid_len

#     # ── video setup ───────────────────────────────────────────────────────
#     title = cur_instruction.strip().replace(" ", "_")

#     output_file_path = os.path.join(save_folder_path, title  + ".png")

#     if trajectory_len > 1:   
#         output_file_path = os.path.join(save_folder_path, title  + ".avi")
#         fourcc = cv2.VideoWriter_fourcc(*"MJPG")
#         video_writer = cv2.VideoWriter(output_file_path, fourcc, 30, (grid_w, grid_h))

#     for step_idx in tqdm(range(trajectory_len), desc="combining the frames"):
#         grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
#         for env_idx in range(total_env):
#             row = env_idx // env_grid_len
#             col = env_idx  % env_grid_len
#             grid[row*h:(row+1)*h, col*w:(col+1)*w] = all_trajectories[env_idx][step_idx]

#         print(f"[DEBUG INFO generate_sample_task_video] grid size {len(grid[0])} x {len(grid)}")

#         if trajectory_len > 1:    
#             video_writer.write(grid)
#         else:
#             print(f"[DEBUG INFO generate_sample_task_video]： trajectory_len=1, just print a single image")
#             cv2.imwrite(output_file_path, grid)

#     if trajectory_len > 1:   
#         video_writer.release()


