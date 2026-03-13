"""
Here we have an object class: bddl generator.
Each bddl generator generate for a specific type of subtasks.
"""

from abc import ABC, abstractmethod
from typing import Optional



from .variables import _PROBLEM_CLASS, OBJECT_NUM_LIMITS, OBJECT_POOL, BOWL_TYPE, INSTRUCTION_TEMPLATES
from .env_generate_utils import allocate_obj_to_region, parse_cell_region


import os
import re
import random 
import tempfile
import textwrap
import numpy as np


"""
If input language is none, a random instruction is picked
"""


def generate_bddl(
        resolved_language: str,
        inst2region,
        regions,
        all_objects,
        target_object: str,
        bowl_instance: str=f"{BOWL_TYPE}_0", # assume only one bowl if not provided
        save_bddl: bool=True,
        output_path: str=None

):
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
        "target_place":          bowl_instance, 
        "target_region":         inst2region[target_object]
    }


def generate_middle_pick_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,  
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    
    # the number of objects are between 3, 5, 7
    rng = np.random.RandomState(seed)  # works for both None and int seeds

    obj_num = rng.choice([3, 5, 7])

    
    # objects to be passed to allocate regions
    obj_list = []



    obj_count_dict = {}
    for _ in range(obj_num):
        cur_obj_type = random.choice(OBJECT_POOL) if seed is not None else rng.choice(OBJECT_POOL)
        if cur_obj_type in obj_count_dict.keys():
            obj_count_dict[cur_obj_type] += 1
        else:
            obj_count_dict[cur_obj_type] = 0

        obj_list.append(f"{cur_obj_type}_{obj_count_dict[cur_obj_type]}")
    

    # allocate the object to each region
    inst2region, regions = allocate_obj_to_region(obj_list,
                                                has_bowl=True,
                                                grid_size=grid_size,
                                                seed=seed,
                                                need_middle_object=True # need a middle object
                                                )
    
    # need to sort the objects by their x-coordinate (row index) to determine the target object based on the language instruction
    # sort inst2region by row index (x-coordinate) in descending order (largest x = closest to camera)
    sorted_objects = sorted(inst2region.keys(), key=lambda obj: parse_cell_region(inst2region[obj])[0], reverse=True)

    # get the middle object without the bowl
    sorted_objects = [obj for obj in sorted_objects if BOWL_TYPE not in obj]


    print(f"[DEBUG INFO generate_middle_pick_task]: sorted object types {sorted_objects}")
    middle_object = sorted_objects[len(sorted_objects) // 2]

    # generate bddl based on input
    return generate_bddl(resolved_language=language, 
                         inst2region=inst2region, 
                         regions=regions, 
                         all_objects=obj_list, 
                         target_object=middle_object, 
                         save_bddl=save_bddl, 
                         output_path=output_path)


    


    

def generate_random_rank_task_bddl(
    language: str = random.choice(INSTRUCTION_TEMPLATES),
    seed: Optional[int] = None,
    num_objects: int = 10,
    grid_size: Optional[int] = 20,  
    object_pool: Optional[list] = OBJECT_POOL,
    bowl_type: str = BOWL_TYPE,
    object_types: str = None,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    
    """
    Parse the instruction to determine what type of bddl to be generated
    """
    lang = language.lower().strip()


    # ── middle pick ───────────────────────────────────────────────────────────
    if "middle" in lang:
        # generate 
        return generate_middle_pick_task_bddl(language, seed, grid_size=grid_size, extra_regions=extra_regions, output_path=output_path, save_bddl=save_bddl)

    # ── feature pick (largest / smallest) ────────────────────────────────────
    if re.search(r"\b(largest|smallest)\b", lang):
        print(f"[DEBUG INFO]：{language}, feature pick task is not implemented")
        return


    # ── allocentric place: "bowl closest/furthest to it" ─────────────────────
    if re.search(r"bowl.*(closest|furthest|furtherest|farthest).*\bit\b", lang):
        print(f"[DEBUG INFO]：{language}, allocentric place task is not implemented")
        return

    # ── allocentric pick: "object closest/furthest to the bowl" ──────────────
    if re.search(r"pick the (?:\d+(?:st|nd|rd|th)?\s+)?object\s+(?:closest|furthest|furtherest|farthest)\s+to the bowl", lang):
        print(f"[DEBUG INFO]：{language}, allocentric pick task is not implemented")
        return

    # ── egocentric place: "place in the [rank] bowl" ─────────────────────────
    if re.search(r"place in the (?:\d+(?:st|nd|rd|th)?\s+)?(?:closest|furthest|furtherest|farthest)\s+bowl\s*\.", lang):
        print(f"[DEBUG INFO]：{language}, egocentric place task is not implemented")
        return

     # ── egocentric pick: "pick the [rank] object" ────────────────────────────
    if re.search(r"pick the.*(closest|furthest|furtherest|farthest|\d+\w*)\s+object", lang):
        print(f"[DEBUG INFO]：{language}, egocentric pick task is not implemented")
        return



    

