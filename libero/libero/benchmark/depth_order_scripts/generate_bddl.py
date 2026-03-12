import os
import re
import sys
import tempfile
import textwrap
import numpy as np

from typing import Optional


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..","..", "LIBERO"))



"""
A script to generate BDDL files from a given domain and problem definition. The BDDL files will be used for benchmarking the performance of different planners on the specified domain and problem. The script takes in the domain and problem definitions, processes them, and outputs the corresponding BDDL files in a specified directory.

All objects except the bowl willbe of same type.

Table is split into i by i cells, i determined by the number of objects.
"""

_PROBLEM_CLASS = "LIBERO_TABLETOP_MANIPULATION"


OBJECT_POOL = [
    "akita_black_bowl",
    "plate",
    "cookies",
    "milk",
    "moka_pot",
    "glazed_rim_porcelain_ramekin",
    "tomato_sauce",
    "alphabet_soup",
    "butter",
    "ketchup",
    "orange_juice",
]

# maximum number of objects of each type to include in the task (including the bowl)
OBJECT_NUM_LIMITS = {
    "plate": 5,
    "cookies": 7,
    "milk": 10,
    "moka_pot": 10,
    "glazed_rim_porcelain_ramekin": 10,
    "tomato_sauce": 10,
    "alphabet_soup": 10,
    "butter": 10,
    "ketchup": 10,
    "orange_juice": 10,
}

# number of cells that must be between two plates (in any direction) to avoid overlap
OBJECT_SPACING_REQUIREMENTS = {
    "akita_black_bowl": 6,
    "plate": 7, 
    "cookies": 5,
    "milk": 3,
    "moka_pot": 4,
    "glazed_rim_porcelain_ramekin": 4,
    "tomato_sauce": 3,
    "alphabet_soup": 3,
    "butter": 4,
    "ketchup": 3,
    "orange_juice": 3,
}

BOWL_TYPE = "akita_black_bowl"



def make_table_regions(n: int) -> dict:
    """
    Generate an n×n grid of table regions.
    e.g. n=4 → 16 cells, n=5 → 25 cells.
    """
    regions = {}
    xs = np.linspace(-0.20, 0.20, n + 1)
    ys = np.linspace(-0.25, 0.25, n + 1)
    for i, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        for j, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            regions[f"cell_{i}_{j}"] = (float(x0), float(y0), float(x1), float(y1))
    return regions


def _parse_ranking_index(language: str, num_items: int) -> int:
    """
    Parse the language instruction to get a 0-based ranking index.
    Ranking is by ascending x (closest = smallest x = closest to camera).
    """
    lang = language.lower()
    if re.search(r"\bclosest\b", lang) and not re.search(r"\d", lang):
        return 0
    if re.search(r"\bfurthest\b|\bfurtherest\b|\bfarthest\b", lang) and not re.search(r"\d", lang):
        return -1

    # "2nd closest", "3rd closest", etc.
    m = re.search(r"(\d+)(?:st|nd|rd|th)?\s+closest", lang)
    if m:
        return int(m.group(1)) - 1

    # "2nd furthest", "3rd furthest", etc.
    m = re.search(r"(\d+)(?:st|nd|rd|th)?\s+(?:furthest|furtherest|farthest)", lang)
    if m:
        # 1st furthest = index -1, 2nd furthest = index -2, ...
        return -int(m.group(1))

    raise ValueError(
        f"Cannot parse ranking from language: '{language}'. "
        "Use 'closest', 'furthest', or ordinals like '2nd closest'."
    )


def prep_for_display(img):
    bgr = img[..., ::-1]
    bgr = np.flipud(bgr)
    return bgr


"""
After th survey we have 4 main task categories which we can consider.

1. Egocentric_pick_and_place: pick item position relative to main camera 
    - pick the closest butter and place in the bowl.
2. Allocentric_pick_and_place: pick item position relative to another object
    - pick the butter closest to the bowl and place in the bowl.
3. depth_order_place: place item on object of specific ordinal rank
    - pick the butter and place in the 3rd closest bowl.
4. Order by feature: Pick item based on specific features.
    - pick the 2nd largest item and place in the bowl.

Each task is a simple pick and place: need to know what object to pick and where to place.
For data collection, we need the specific locations and how to interact in order to perform data collection
"""


def generate_random_depth_order_task_bddl(
    language: str = "Place the closest object in the bowl.",
    seed: Optional[int] = None,
    num_objects: int = 2,
    grid_size: Optional[int] = 20,  
    object_pool: Optional[list] = None,
    bowl_type: str = BOWL_TYPE,
    object_types: str = None,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    pass



def generate_random_bddl(
    language: str = "Place the closest object in the bowl.",
    seed: Optional[int] = None,
    num_objects: int = 2,
    grid_size: Optional[int] = 20,  
    object_pool: Optional[list] = None,
    bowl_type: str = BOWL_TYPE,
    object_types: str = None,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    '''
    This is the generation of the bddl task file for the tabletop manipulation task. 
    The function takes in a language instruction, a random seed for reproducibility,
    the number of distractor objects to include, 
    an optional object pool to sample from, 
    the type of bowl to use, 
    an optional output path to save the generated BDDL file, 
    and any extra regions to include in the BDDL file.
    '''
    # ── get object types and regions ──────────────────────────────
   

    rng     = np.random.RandomState(seed) # Create a random number generator with the given seed for reproducibility
    pool    = list(object_pool or OBJECT_POOL) # Use the provided object pool or the default one
    non_bowl_pool    = [obj for obj in pool if obj != bowl_type] # Remove the bowl type from the pool to get the non-bowl objects
    regions = {**make_table_regions(grid_size), **(extra_regions or {})}

    # Ensure there are non-bowl objects to sample from
    if not non_bowl_pool:
        raise ValueError("object_pool must contain at least one non-bowl type.")
    
    # get the object types for the task
    if object_types is None:
        object_types = rng.choice(non_bowl_pool)
    # After object_types is determined, substitute into language if it's generic
    resolved_language = language.replace("object", object_types).replace("item", object_types)

    print(f"DEBUG INFO: {resolved_language}")

    # determine the number of objects to include based on the spacing requirements 
    # and limits for the chosen object type
    object_spacing = OBJECT_SPACING_REQUIREMENTS[object_types]
    if OBJECT_NUM_LIMITS[object_types] < num_objects:
        num_objects = OBJECT_NUM_LIMITS[object_types]

    item_instances = [f"{object_types}_{i}" for i in range(num_objects)]
    bowl_instance  = f"{bowl_type}_0"

    # all objects in the task, including the bowl and the distractor objects
    all_objects = item_instances + [bowl_instance,]


    # ── assign non-overlapping placement regions ──────────────────────────────

    def parse_cell_region(region_name):
        # region_name is in the format "cell_i_j"
        _, i, j = region_name.split("_")
        return int(i), int(j)

    def remove_regions(occupied_region_name, spacing: int = 1):
            """
            Remove all regions in the same row as the occupied region (enforcing one object per row),
            plus all regions within `spacing` cells in any direction around the occupied region.
            
            spacing=1 → removes immediate neighbours (original behaviour)
            spacing=2 → removes 2-cell radius around occupied cell
            spacing=0 → removes only the same row (no extra buffer)
            """
            occupied_row, occupied_col = parse_cell_region(occupied_region_name)
            
            # Remove entire row (enforces one object per x-depth level)
            to_remove = [r for r in region_names if parse_cell_region(r)[0] == occupied_row]
            for r in to_remove:
                if r in region_names:
                    region_names.remove(r)

            # Remove all cells within `spacing` distance in both axes
            for dx in range(-spacing, spacing + 1):
                for dy in range(-spacing, spacing + 1):
                    if dx == 0 and dy == 0:
                        continue  # already removed via row sweep above
                    adj = f"cell_{occupied_row + dx}_{occupied_col + dy}"
                    if adj in region_names:
                        region_names.remove(adj)

    MAX_RETRIES = 100

    for attempt in range(MAX_RETRIES):
        region_names = list(regions.keys())  # reset available regions each attempt
        inst2region = {}

        try:
            # assign bowl first
            bowl_region = rng.choice(region_names)
            region_names.remove(bowl_region)
            remove_regions(bowl_region, spacing=OBJECT_SPACING_REQUIREMENTS[bowl_type])

            # assign each item to a unique row with spacing
            for obj in item_instances:
                if not region_names:
                    raise ValueError("No regions left for objects.")
                inst2region[obj] = rng.choice(region_names)
                remove_regions(inst2region[obj], spacing=object_spacing)

            break  # success

        except (ValueError, IndexError):
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Could not place all {len(item_instances)} objects after "
                    f"{MAX_RETRIES} attempts. Try reducing num_objects, spacing, or increasing grid_size."
                )
            continue  # retry with new rng choices

    # FOR DEBUGGING: print the final object to region assignments
    print("Final object to region assignments:")
    for obj, region in inst2region.items():
        print(f"  {obj} → {region}, occupies row {parse_cell_region(region)[0]}")

    # ── determine target object based on language instruction ──────────────────────────────
    try:
        target_index = _parse_ranking_index(language, num_objects)
    except ValueError as e:
        print(f"Error parsing language instruction: {e}")
        return {}
    
    # need to sort the objects by their x-coordinate (row index) to determine the target object based on the language instruction
    # sort inst2region by row index (x-coordinate) in descending order (largest x = closest to camera)
    sorted_objects = sorted(inst2region.keys(), key=lambda obj: parse_cell_region(inst2region[obj])[0], reverse=True)
    target_object = sorted_objects[target_index] # get the target object based on the parsed ranking   

    # IMPORTANT: add the bowl region assignment to inst2region only after determining the target object, since the bowl's position also affects the ranking of the objects
    inst2region[bowl_instance] = bowl_region
    
    # DEBUGGING: print the sorted objects and the target object
    print(f"Sorted objects by row index (closest to furthest): {sorted_objects}")
    print(f"current language instruction: '{language}', hence the rank index is {target_index} and the target object is '{target_object}'")
    

    ####### CONSTRUCTION OF THE BDDL FILE CONTENTS WOULD GO HERE #######

    # ── :regions block ────────────────────────────────────────────────────────
    # Now we need to construct the :regions block of the BDDL file based on the assigned regions for each object. 
    # Each region will be defined by its bounding box coordinates (x0, y0, x1, y1) corresponding to the 
    # assigned cell on the table. We will also include any extra regions provided in the input.
    region_lines = []
    for rname, (x0, y0, x1, y1) in regions.items():
        region_lines.append(
            f"    ({rname}\n"
            f"        (:target main_table)\n"
            f"        (:ranges (\n"
            f"            ({x0:.4f} {y0:.4f} {x1:.4f} {y1:.4f})\n"
            f"          )\n"
            f"        )\n"
            f"    )"
        )

    # ── :objects block ───────────────────────────────────────────────────────
    # Next, we construct the :objects block of the BDDL file. This block will list all the objects 
    # in the task, including the bowl and the distractor objects. Each object will be defined by its 
    # type and its initial region (the cell it is placed in). We will also include any extra objects 
    # provided in the input.
    type_to_insts: dict = {}
    for inst in all_objects:
        obj_type = inst.rsplit("_", 1)[0] # get the type by removing the instance index (e.g. "plate_0" → "plate")
        if obj_type not in type_to_insts:
            type_to_insts[obj_type] = []
        type_to_insts[obj_type].append(inst)

    obj_lines = [f"    {' '.join(v)} - {k}" for k, v in type_to_insts.items()]


    # ── :init block ───────────────────────────────────────────────────────────
    init_lines = [
        f"    (On {inst} main_table_{rname})"
        for inst, rname in inst2region.items()
    ]

        # ── assemble ──────────────────────────────────────────────────────────────
    NL = "\n"
    bddl = textwrap.dedent(f"""\
        (define (problem {_PROBLEM_CLASS})
          (:domain robosuite)
          (:language {resolved_language})
          (:regions
        {NL.join(region_lines)}
          )
          (:fixtures
            main_table - table
          )
          (:objects
        {NL.join(obj_lines)}
          )
          (:obj_of_interest
            {target_object}
            {bowl_instance}
          )
          (:init
        {NL.join(init_lines)}
          )
          (:goal
            (And (On {target_object} {bowl_instance}))
          )
        )
    """)

    


    
    # find the location of the target object (center of its assigned region) to be used as the target location for evaluation
    bounds = regions[inst2region[target_object]]
    target_region_center = (
        (bounds[0] + bounds[2]) / 2.0,
        (bounds[1] + bounds[3]) / 2.0,
    )

    if save_bddl:
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".bddl", prefix="libero_random_")
            os.close(fd)

        with open(output_path, "w") as f:
            f.write(bddl)

        return {
        "bddl": bddl,
        "bddl_path":             os.path.abspath(output_path),
        "target_object":         target_object,
        "target_region":         inst2region[target_object],
        "target_region_center":  target_region_center,
    }





if __name__ == "__main__":
    from random import randint
    import numpy as np
    import cv2
    from tqdm import tqdm
    from libero.libero.envs import OffScreenRenderEnv
    # from shuijie_codebase.env.generate_bddl import generate_random_bddl

    import cv2
    from tqdm import tqdm
    from random import randint
    import numpy as np

    # # Video writer setup
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # fps = 1
    # frame_size = (512, 256)  # side-by-side: agentview + eye_in_hand
    # num_of_env = 25


    # for cur_obj_types in OBJECT_POOL[1:]:
        
    #     video_path = f"./shuijie_codebase/video/{cur_obj_types}_env_rollout.mp4"
    #     video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)


    #     for i in tqdm(range(num_of_env), desc="Running 100 different envs to check for error"):
    #         seed = randint(0, 1000)
    #         language = "Place the object 3rd closest to the main camera in the bowl."

    #         result = generate_random_bddl(language=language, seed=seed, num_objects=10, object_types=cur_obj_types)

    #         env = OffScreenRenderEnv(
    #             bddl_file_name=result["bddl_path"],
    #             robots=["Panda"],
    #             camera_heights=256,
    #             camera_widths=256,
    #         )

    #         env.seed(seed)
    #         obs = env.reset()

    #         try:
    #             agentview = prep_for_display(obs["agentview_image"]) if "agentview_image" in obs else np.zeros((256, 256, 3), dtype=np.uint8)
    #             eye_in_hand = prep_for_display(obs["robot0_eye_in_hand_image"]) if "robot0_eye_in_hand_image" in obs else np.zeros((256, 256, 3), dtype=np.uint8)

    #             # Combine side by side and write to video
    #             combined = np.concatenate([agentview, eye_in_hand], axis=1)  # (256, 512, 3)
    #             video_writer.write(combined)
    #         except Exception as e:
    #             print(f"[{i}] Frame write failed: {e}")

    #         env.close()

    #     video_writer.release()
    #     print(f"Video saved to {video_path}")


    seed = randint(0, 1000)
    language = "Place the object 3rd closest to the camera in the bowl."

    result = generate_random_bddl(language=language, seed=seed, num_objects=10)


    print(result)

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