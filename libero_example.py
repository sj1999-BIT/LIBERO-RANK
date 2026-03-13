import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm

import cv2
import numpy as np


benchmark_dict = benchmark.get_benchmark_dict()
print(benchmark_dict)
task_suite_name = "libero_rank" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}

def prep_for_display(img):
    bgr = img[..., ::-1]
    bgr = np.flipud(bgr)
    return bgr



env = OffScreenRenderEnv(**env_args)
env.seed(0)

obs = env.reset()

target_key = "akita_black_bowl_0"

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