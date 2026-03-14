from libero.libero.benchmark.rank_scripts.bddl_generator import generate_random_rank_task_bddl
from random import randint
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys


libero_path =  os.path.join(os.path.dirname(__file__), "..","..", "..", "..", "..")
print(f"libero path: {libero_path}")
sys.path.insert(0, libero_path)


import cv2
from tqdm import tqdm
from libero.libero.envs import OffScreenRenderEnv
# from shuijie_codebase.env.generate_bddl import generate_random_bddl

import cv2
from tqdm import tqdm
from random import randint
import numpy as np




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

def generate_sample_task_video(instruction,
                               trajectory_len=100,
                               save_video=False, 
                               render_video=False, 
                               env_grid_len=1):
    total_env = env_grid_len * env_grid_len
    
     # ── collect frames from each trajectory ──────────────────────────────────
    all_trajectories = []  # list of lists of frames, one per env

    for env_idx in range(total_env):
        
        result = generate_random_rank_task_bddl(language=instruction, num_objects=10, save_bddl=True)

        actual_instruction = result["resolved_language"]

        seed = randint(0, 1000)

        env = OffScreenRenderEnv(bddl_file_name=result["bddl_path"], robots=["Panda"],
                                camera_heights=256, camera_widths=256)
        env.seed(seed)
        obs = env.reset()

        target_key = result["target_object"]  # e.g. "cookies_1"

        target_pos  = obs[f"{target_key}_pos"]  # actual sim position (x, y, z)
        target_quat = obs[f"{target_key}_quat"]

        frames = []

        for step in tqdm(range(trajectory_len), desc="Moving to target"):
            robot_eef_pos = obs["robot0_eef_pos"]

            # Use actual target_pos from obs — update each step in case of physics drift
            target_pos = obs[f"{target_key}_pos"]
            delta_pos  = target_pos - robot_eef_pos# keep height constant

            print(delta_pos)

            if sum(delta_pos[:2]) < 0.03:
                # move to the target bowl after reaching the target pick
                target_key = result["target_place"]

            action_7dim = np.zeros(7)
            action_7dim[:3] = np.clip(delta_pos * 10, -1, 1)
            action_7dim[2] = 0.0

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

        all_trajectories.append(frames)

        env.close()
        cv2.destroyAllWindows()

    if save_video:

        
        # combine frames to grid
        h, w = all_trajectories[0][0].shape[:2]
        grid_h = h * env_grid_len
        grid_w = w * env_grid_len

        # ── video setup ───────────────────────────────────────────────────────
        video_name = cur_instruction.strip().replace(" ", "_") + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(video_name, fourcc, 30, (grid_w, grid_h))

        for step_idx in tqdm(range(trajectory_len), desc="combining the frames"):
            grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            for env_idx in range(total_env):
                row = env_idx // env_grid_len
                col = env_idx  % env_grid_len
                grid[row*h:(row+1)*h, col*w:(col+1)*w] = all_trajectories[env_idx][step_idx]

            print(f"[DEBUG INFO generate_sample_task_video] grid size {len(grid[0])} x {len(grid)}")

            video_writer.write(grid)

        video_writer.release()





if __name__ == "__main__":

    for cur_instruction in tqdm(INSTRUCTION_TEMPLATES, desc="going through the instructions"):
        generate_sample_task_video(cur_instruction, save_video=True, render_video=True, env_grid_len=2)
        # # # ── video setup ───────────────────────────────────────────────────────
        # # video_name = cur_instruction.strip().replace(" ", "_") + ".mp4"
        # # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # # video_writer = cv2.VideoWriter(video_name, fourcc, 30, (256, 256))
        
        # result = generate_random_rank_task_bddl(language=cur_instruction, num_objects=10, save_bddl=True)

        # seed = randint(0, 1000)

        # # print(result)

        # env = OffScreenRenderEnv(bddl_file_name=result["bddl_path"], robots=["Panda"],
        #                         camera_heights=256, camera_widths=256)
        # env.seed(seed)
        # obs = env.reset()

        # # print(obs)

        # # ── get ACTUAL target position from obs, not region center ────────────────
        # # target_key = result["target_place"]  # e.g. "cookies_1"
        # target_key = result["target_object"]  # e.g. "cookies_1"

        # target_pos  = obs[f"{target_key}_pos"]  # actual sim position (x, y, z)
        # target_quat = obs[f"{target_key}_quat"]

        # if "agentview_image" in obs:
        #     filepath = f"rank_img_results/{create_title(cur_instruction)}.png"
        #     print(filepath)
        #     cv2.imwrite(filepath, prep_for_display(obs["agentview_image"]))




        # for step in tqdm(range(500), desc="Moving to target"):
        #     robot_eef_pos = obs["robot0_eef_pos"]

        #     # Use actual target_pos from obs — update each step in case of physics drift
        #     target_pos = obs[f"{target_key}_pos"]
        #     delta_pos  = target_pos[:2] - robot_eef_pos[:2]# keep height constant

        #     # print(sum(delta_pos))

        #     if sum(delta_pos) < 0.02:
        #         # move to the target bowl after reaching the target pick
        #         target_key = result["target_place"]
                

        #     action_7dim = np.zeros(7)
        #     action_7dim[:2] = np.clip(delta_pos * 10, -1, 1)
        #     action_7dim[6]  = -1.0   # gripper open

        #     obs, reward, done, info = env.step(action_7dim)

        #     try:
        #         if "agentview_image" in obs:
        #             cv2.imshow("Main Camera", prep_for_display(obs["agentview_image"]))
        #         if "robot0_eye_in_hand_image" in obs:
        #             cv2.imshow("Gripper Camera", prep_for_display(obs["robot0_eye_in_hand_image"]))
        #         if cv2.waitKey(1) & 0xFF == 27:
        #             break
        #     except Exception:
        #         pass

        # env.close()
        # cv2.destroyAllWindows()