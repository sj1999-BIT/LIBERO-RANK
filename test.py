from libero.libero.benchmark.depth_order_scripts.bddl_generator import generate_random_depth_order_task_bddl
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
INSTRUCTION_TEMPLATES = {
    # # egocentric pick tasks: many same object + 1 bowl
    # "Pick the closest object and place in the bowl.",
    # "Pick the furtherest object and place in the bowl.",
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
    # # egocentric place tasks: 1 object + many bowls
    # "Pick the object and place in the closest bowl.",
    # "Pick the object and place in the furtherest bowl.",
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
    # # allocentric pick: many same object + 1 bowl
    # "Pick the object closest to the bowl and place in the bowl.",
    # "Pick the object furtherest to the bowl and place in the bowl.",
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
    # # allocentric place: 1 object + many bowls
    # "Pick the object and place in the bowl closest to it.",
    # "Pick the object and place in the bowl furtherest from it.",
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
    # # pick by feature: many different object + 1 bowl
    # "Pick the largest object and place in the bowl.",
    # "Pick the smallest object and place in the bowl.",
    # "Pick the 1st largest object and place in the bowl.",
    # "Pick the 2nd largest object and place in the bowl.",
    # "Pick the 3rd largest object and place in the bowl.",
    # "Pick the 4th largest object and place in the bowl.",
    # "Pick the 5th largest object and place in the bowl.",
    # "Pick the 6th largest object and place in the bowl.",
    # "Pick the 7th largest object and place in the bowl.",
    # "Pick the 1st smallest object and place in the bowl.",
    # "Pick the 2nd smallest object and place in the bowl.",
    # "Pick the 3rd smallest object and place in the bowl.",
    # "Pick the 4th smallest object and place in the bowl.",
    # "Pick the 5th smallest object and place in the bowl.",
    # "Pick the 6th smallest object and place in the bowl.",
    # "Pick the 7th smallest object and place in the bowl.",
    # middle pick: 3 different object with col restriction + one bowl
    "Pick the object in the middle and place in the bowl."
}


def prep_for_display(img):
    bgr = img[..., ::-1]
    bgr = np.flipud(bgr)
    return bgr


if __name__ == "__main__":

    for cur_instruction in INSTRUCTION_TEMPLATES:
        result = generate_random_depth_order_task_bddl(language=cur_instruction, num_objects=10, save_bddl=True)

        seed = randint(0, 1000)

        # print(result)

        env = OffScreenRenderEnv(bddl_file_name=result["bddl_path"], robots=["Panda"],
                                camera_heights=256, camera_widths=256)
        env.seed(seed)
        obs = env.reset()

        # ── get ACTUAL target position from obs, not region center ────────────────
        target_key = result["target_object"]  # e.g. "cookies_1"
        target_pos  = obs[f"{target_key}_pos"]  # actual sim position (x, y, z)
        target_quat = obs[f"{target_key}_quat"]

        for step in tqdm(range(500), desc="Moving to target"):
            robot_eef_pos = obs["robot0_eef_pos"]

            # Use actual target_pos from obs — update each step in case of physics drift
            target_pos = obs[f"{target_key}_pos"]
            delta_pos  = target_pos[:2] - robot_eef_pos[:2]# keep height constant

            action_7dim = np.zeros(7)
            action_7dim[:2] = np.clip(delta_pos * 10, -1, 1)
            action_7dim[6]  = -1.0   # gripper open

            obs, reward, done, info = env.step(action_7dim)

            try:
                if "agentview_image" in obs:
                    cv2.imshow("Main Camera", prep_for_display(obs["agentview_image"]))
                if "robot0_eye_in_hand_image" in obs:
                    cv2.imshow("Gripper Camera", prep_for_display(obs["robot0_eye_in_hand_image"]))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            except Exception:
                pass

        env.close()
        cv2.destroyAllWindows()