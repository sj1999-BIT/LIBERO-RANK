"""
Here we have an object class: bddl generator.
Each bddl generator generate for a specific type of subtasks.
"""

from abc import ABC, abstractmethod
from typing import Optional



from .variables import _PROBLEM_CLASS, OBJECT_NUM_LIMITS, OBJECT_POOL, BOWL_TYPE, INSTRUCTION_TEMPLATES, OBJECT_SIZE_RANK
from .env_generate_utils import allocate_obj_to_region, parse_cell_region, parse_ranking_index


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
    object_pool: Optional[list] = OBJECT_POOL,
    bowl_type: str = BOWL_TYPE,
    obj_num: int = 5,       
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
        cur_obj_type = rng.choice(object_pool)
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
                                                need_middle_object=True, # need a middle object
                                                bowl_type=bowl_type
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


def generate_middle_place_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,  
    output_path: Optional[str] = None,
    bowl_type: str = BOWL_TYPE,
    obj_num: int = 5,              
    object_pool: Optional[list] = OBJECT_POOL,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    
    # the number of objects are between 3, 5, 7
    rng = np.random.RandomState(seed)  # works for both None and int seeds


    bowl_num = rng.choice([3, 5, 7])

    
    # a single random object to be picked
    target_obj_type = f"{rng.choice(object_pool)}_0"
    obj_list = [target_obj_type, ]

    for i in range(bowl_num):
        obj_list.append(f"{bowl_type}_{i}")
    

    # allocate the object to each region
    inst2region, regions = allocate_obj_to_region(obj_list,
                                                has_bowl=False, # many bowl
                                                grid_size=grid_size,
                                                seed=seed,
                                                need_middle_object=True, # need a middle bowl
                                                )
    
    # need to sort the objects by their x-coordinate (row index) to determine the target object based on the language instruction
    # sort inst2region by row index (x-coordinate) in descending order (largest x = closest to camera)
    sorted_objects = sorted(inst2region.keys(), key=lambda obj: parse_cell_region(inst2region[obj])[0], reverse=True)

    # get the middle bowl
    sorted_objects = [obj for obj in sorted_objects if bowl_type in obj]


    print(f"[DEBUG INFO generate_middle_place_task_bddl]: sorted object types {sorted_objects}")
    middle_bowl = sorted_objects[len(sorted_objects) // 2]

    # generate bddl based on input
    return generate_bddl(resolved_language=language, 
                         inst2region=inst2region, 
                         regions=regions, 
                         all_objects=obj_list, 
                         target_object=target_obj_type, 
                         bowl_instance=middle_bowl, # bowl instance is the middle one
                         save_bddl=save_bddl, 
                         output_path=output_path)




def generate_egocentric_pick_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    bowl_type: str = BOWL_TYPE,
    object_pool: Optional[list] = OBJECT_POOL,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    
    # the number of objects are between 3, 5, 7
    rng = np.random.RandomState(seed)  # works for both None and int seeds

    
    # objects to be passed to allocate regions
    obj_list = []

    # pick a specifc object type
    object_types = rng.choice(object_pool)


    # After object_types is determined, substitute into language if it's generic
    resolved_language = language.replace("object", object_types).replace("item", object_types)


    obj_num = min(len(object_pool), obj_num)

     
    for index in range(obj_num):
        obj_list.append(f"{object_types}_{index}")
    

    # allocate the object to each region
    inst2region, regions = allocate_obj_to_region(obj_list,
                                                has_bowl=True,
                                                grid_size=grid_size,
                                                seed=seed,
                                                need_middle_object=False, 
                                                bowl_type=bowl_type
                                                )
    
    # need to sort the objects by their x-coordinate (row index) to determine the target object based on the language instruction
    # sort inst2region by row index (x-coordinate) in descending order (largest x = closest to camera)
    sorted_objects = sorted(inst2region.keys(), key=lambda obj: parse_cell_region(inst2region[obj])[0], reverse=True)

    # get the middle object without the bowl
    sorted_objects = [obj for obj in sorted_objects if bowl_type not in obj]

    # get the target rank
    rank = parse_ranking_index(resolved_language, obj_num)

    # print(f"[DEBUG INFO generate_egocentric_pick_task_bddl] instruction {resolved_language}: target rank is {rank}")

    # print(f"[DEBUG INFO generate_egocentric_pick_task_bddl] sorted_objects {sorted_objects}")
    # get the target object
    target_object = sorted_objects[rank]


    # generate bddl based on input
    return generate_bddl(resolved_language=resolved_language, 
                         inst2region=inst2region, 
                         regions=regions, 
                         all_objects=obj_list, 
                         target_object=target_object, 
                         save_bddl=save_bddl, 
                         output_path=output_path)


def generate_egocentric_place_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    object_pool: Optional[list] = OBJECT_POOL,
    bowl_type: str = BOWL_TYPE,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    
    # the number of objects are between 3, 5, 7
    rng = np.random.RandomState(seed)  # works for both None and int seeds


    # cannot have too many bowls
    obj_num = min(OBJECT_NUM_LIMITS[bowl_type], obj_num)

    
    # objects to be passed to allocate regions
    obj_list = []

    # pick a specifc object type
    object_type = rng.choice(object_pool)

    target_pick_obj = f"{object_type}_0"    
    
    # add a single object to the list
    obj_list.append(target_pick_obj)

    # After object_types is determined, substitute into language if it's generic
    resolved_language = language.replace("object", object_type).replace("item", object_type)
     
    for index in range(obj_num):
        obj_list.append(f"{bowl_type}_{index}")
    
    # allocate the object to each region
    inst2region, regions = allocate_obj_to_region(obj_list,
                                                has_bowl=False, # for placing tasks we already add in many bowls
                                                grid_size=grid_size,
                                                seed=seed,
                                                need_middle_object=False, # need a middle object
                                                bowl_type=bowl_type
                                                )
    
    # need to sort the objects by their x-coordinate (row index) to determine the target object based on the language instruction
    # sort inst2region by row index (x-coordinate) in descending order (largest x = closest to camera)
    sorted_objects = sorted(inst2region.keys(), key=lambda obj: parse_cell_region(inst2region[obj])[0], reverse=True)

    # sort the bowls
    sorted_objects = [obj for obj in sorted_objects if BOWL_TYPE in obj]

    # get the target rank
    rank = parse_ranking_index(resolved_language, obj_num)

    # print(f"[DEBUG INFO generate_egocentric_pick_task_bddl] instruction {language}: target rank is {rank}")

    # get the target object
    target_bowl_instance = sorted_objects[rank]


    # generate bddl based on input
    return generate_bddl(resolved_language=resolved_language, 
                         inst2region=inst2region, 
                         regions=regions, 
                         all_objects=obj_list, 
                         target_object=f"{object_type}_0", 
                         bowl_instance=target_bowl_instance, # specific the bowl to be placeed
                         save_bddl=save_bddl, 
                         output_path=output_path)


def generate_allocentric_pick_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    bowl_type: str = BOWL_TYPE,
    object_pool: Optional[list] = OBJECT_POOL,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl:bool = False
) -> dict:
    
    # the number of objects are between 3, 5, 7
    rng = np.random.RandomState(seed)  # works for both None and int seeds

    
    # objects to be passed to allocate regions
    obj_list = []

    # pick a specifc object type
    object_types = rng.choice(object_pool)


    # After object_types is determined, substitute into language if it's generic
    resolved_language = language.replace("object", object_types).replace("item", object_types)

     
    for index in range(obj_num):
        obj_list.append(f"{object_types}_{index}")
    

    # allocate the object to each region
    inst2region, regions, distance_map = allocate_obj_to_region(obj_list,
                                                has_bowl=True,
                                                grid_size=grid_size,
                                                seed=seed,
                                                need_middle_object=False, 
                                                bowl_type=bowl_type,
                                                need_allocation_dist=True,
                                                allocated_object_type=f"{bowl_type}_0", # need to make sure all objects have different distance to bowl
                                                )
    
    # need to sort the objects by their x-coordinate (row index) to determine the target object based on the language instruction
    # sort inst2region by row index (x-coordinate) in descending order (largest x = closest to camera)
    sorted_objects = sorted([obj for obj in inst2region if bowl_type not in obj], key=lambda obj: distance_map[obj])



    # print(f"[DEBUG INFO generate_allocentric_pick_task_bddl]: sorted object distances obtained:")
    # for cur_obj in sorted_objects:
    #     print(f"[DEBUG INFO generate_allocentric_pick_task_bddl]: {cur_obj} distances {distance_map[cur_obj]}:")

    # get the middle object without the bowl
    sorted_objects = [obj for obj in sorted_objects if bowl_type not in obj]

    # get the target rank
    rank = parse_ranking_index(resolved_language, obj_num)



    # get the target object
    target_object = sorted_objects[rank]

    # print(f"[DEBUG INFO generate_allocentric_pick_task_bddl] instruction {language}: target rank is {rank}")
    # print(f"[DEBUG INFO generate_allocentric_pick_task_bddl] target_object {target_object}: distance is {distance_map[target_object]}")


    # generate bddl based on input
    return generate_bddl(resolved_language=resolved_language, 
                         inst2region=inst2region, 
                         regions=regions, 
                         all_objects=obj_list, 
                         target_object=target_object, 
                         save_bddl=save_bddl, 
                         output_path=output_path)


def generate_allocentric_place_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    bowl_type: str = BOWL_TYPE,
    object_pool: Optional[list] = OBJECT_POOL,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl: bool = False
) -> dict:

    rng = np.random.RandomState(seed)

    # cannot have too many bowls
    obj_num = min(OBJECT_NUM_LIMITS[bowl_type], obj_num)

    # pick a specific non-bowl object type
    object_type = rng.choice(object_pool)
    target_pick_obj = f"{object_type}_0"

    # After object_type is determined, substitute into language if it's generic
    resolved_language = language.replace("object", object_type).replace("item", object_type)

    # one pick object + multiple bowls
    obj_list = [target_pick_obj]
    for index in range(obj_num):
        obj_list.append(f"{bowl_type}_{index}")

    # allocate — sort bowls by distance to the pick object, so all distances must be unique
    inst2region, regions, distance_map = allocate_obj_to_region(
        obj_list,
        has_bowl=False,         # bowls already added manually above
        grid_size=grid_size,
        seed=seed,
        need_middle_object=False,
        bowl_type=bowl_type,
        need_allocation_dist=True,
        allocated_object_type=target_pick_obj,  # measure distances relative to the pick object
    )

    # sort bowls ascending by distance to pick object (index 0 = closest, index -1 = furthest)
    sorted_bowls = sorted(
        [obj for obj in inst2region if bowl_type in obj],
        key=lambda obj: distance_map[obj]
    )

    # print(f"[DEBUG INFO generate_allocentric_place_task_bddl]: bowl distances to '{target_pick_obj}':")
    # for bowl in sorted_bowls:
    #     print(f"[DEBUG INFO generate_allocentric_place_task_bddl]:   {bowl} → {distance_map[bowl]}")

    # get target rank from language
    rank = parse_ranking_index(resolved_language, len(sorted_bowls))
    target_bowl_instance = sorted_bowls[rank]

    # print(f"[DEBUG INFO generate_allocentric_place_task_bddl] instruction '{language}': target rank is {rank}")
    # print(f"[DEBUG INFO generate_allocentric_place_task_bddl] target bowl '{target_bowl_instance}': distance is {distance_map[target_bowl_instance]}")

    return generate_bddl(
        resolved_language=resolved_language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_pick_obj,
        bowl_instance=target_bowl_instance,  # place in the rank-th bowl
        save_bddl=save_bddl,
        output_path=output_path,
    )


def generate_pick_by_feature_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 5,               # now = number of *types* to compare
    bowl_type: str = BOWL_TYPE,
    object_pool: Optional[list] = OBJECT_SIZE_RANK,
    output_path: Optional[str] = None,
    extra_regions: Optional[dict] = None,
    save_bddl: bool = False
) -> dict:
    # the number of objects are between 3, 5, 7
    rng = np.random.RandomState(seed)  # works for both None and int seeds

    
    # objects to be passed to allocate regions
    obj_list = []

    obj_num = min(len(object_pool), obj_num)

    # random pick unqiue objects, add _0 suffix to form valid label
    obj_list = [ f"{i}_0" for i in rng.choice(object_pool, size=obj_num, replace=False)]
    

    # allocate the object to each region
    inst2region, regions = allocate_obj_to_region(obj_list,
                                                has_bowl=True,
                                                grid_size=grid_size,
                                                seed=seed,
                                                need_middle_object=False, 
                                                bowl_type=bowl_type
                                                )
    
    # need to sort the objects by their x-coordinate (row index) to determine the target object based on the language instruction
    # sort inst2region by row index (x-coordinate) in descending order (largest x = closest to camera)
    sorted_objects = sorted(inst2region.keys(), key=lambda obj: parse_cell_region(inst2region[obj])[0], reverse=True)

    # get the middle object without the bowl
    # sorted_objects = [obj for obj in sorted_objects if bowl_type not in obj]

    sorted_objects = sorted(
        [inst for inst in inst2region if bowl_type not in inst],
        key=lambda inst: OBJECT_SIZE_RANK.index(inst.rsplit("_", 1)[0])
    )

    # get the target rank
    rank = parse_ranking_index(language, obj_num)

    # print(f"[DEBUG INFO generate_pick_by_feature_task_bddl] instruction {language}: target rank is {rank}")

    # get the target object
    target_object = sorted_objects[rank]


    # generate bddl based on input
    return generate_bddl(resolved_language=language, 
                         inst2region=inst2region, 
                         regions=regions, 
                         all_objects=obj_list, 
                         target_object=target_object, 
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


    # ── middle place: "place in the bowl in the middle" ──────────────────────
    if "middle" in lang and re.search(r"place.*bowl.*middle|bowl.*middle", lang):
        print(f"[DEBUG INFO]: {language}, middle place task implemented")
        return generate_middle_place_task_bddl(language=language,
                                            seed=seed,
                                            grid_size=grid_size,
                                            obj_num=num_objects,
                                            output_path=output_path,
                                            extra_regions=extra_regions,
                                            object_pool=object_pool,
                                            save_bddl=save_bddl)

    # ── middle pick: "pick the object in the middle" ─────────────────────────
    if "middle" in lang:
        return generate_middle_pick_task_bddl(language=language,
                                            seed=seed,
                                            grid_size=grid_size,
                                            obj_num=num_objects,
                                            output_path=output_path,
                                            extra_regions=extra_regions,
                                            object_pool=object_pool,
                                            save_bddl=save_bddl)
    
    # ── feature pick (largest / smallest) ────────────────────────────────────
    if re.search(r"\b(largest|smallest)\b", lang):
        print(f"[DEBUG INFO]：{language}, feature pick task is implemented")
        return generate_pick_by_feature_task_bddl(language=language,
                                                  seed=seed,
                                                  grid_size=grid_size,
                                                  obj_num=num_objects,
                                                  output_path=output_path,
                                                  extra_regions=extra_regions,
                                                #   object_pool=object_pool, only use predefined objeck with rank
                                                  save_bddl=save_bddl)


    # ── allocentric place: "bowl closest/furthest to it" ─────────────────────
    if re.search(r"bowl.*(closest|furthest|furtherest|farthest).*\bit\b", lang):
        print(f"[DEBUG INFO]：{language}, allocentric place task is implemented")
        return generate_allocentric_place_task_bddl(language=language,
                                            seed=seed,
                                            grid_size=grid_size,
                                            obj_num=num_objects,
                                            output_path=output_path,
                                            extra_regions=extra_regions,
                                            object_pool=object_pool,
                                            save_bddl=save_bddl)

    # ── allocentric pick: "object closest/furthest to the bowl" ──────────────
    if re.search(r"pick the (?:\d+(?:st|nd|rd|th)?\s+)?object\s+(?:closest|furthest|furtherest|farthest)\s+to the bowl", lang):
        print(f"[DEBUG INFO]：{language}, allocentric pick task is implemented")
        return generate_allocentric_pick_task_bddl(language=language,
                                                  seed=seed,
                                                  grid_size=grid_size,
                                                  obj_num=num_objects,
                                                  output_path=output_path,
                                                  extra_regions=extra_regions,
                                                  object_pool=object_pool,
                                                  save_bddl=save_bddl)

    # ── egocentric place: "place in the [rank] bowl" ─────────────────────────
    if re.search(r"place in the (?:\d+(?:st|nd|rd|th)?\s+)?(?:closest|furthest|furtherest|farthest)\s+bowl\s*\.", lang):
        print(f"[DEBUG INFO]：{language}, egocentric place task implemented")
        return generate_egocentric_place_task_bddl(language=language,
                                                  seed=seed,
                                                  grid_size=grid_size,
                                                  obj_num=num_objects,
                                                  output_path=output_path,
                                                  extra_regions=extra_regions,
                                                  object_pool=object_pool,
                                                  save_bddl=save_bddl)

     # ── egocentric pick: "pick the [rank] object" ────────────────────────────
    if re.search(r"pick the.*(closest|furthest|furtherest|farthest|\d+\w*)\s+object", lang):
        print(f"[DEBUG INFO]：{language}, egocentric pick task generated")
        return generate_egocentric_pick_task_bddl(language=language,
                                                  seed=seed,
                                                  grid_size=grid_size,
                                                  obj_num=num_objects,
                                                  output_path=output_path,
                                                  extra_regions=extra_regions,
                                                  object_pool=object_pool,
                                                  save_bddl=save_bddl)



    

