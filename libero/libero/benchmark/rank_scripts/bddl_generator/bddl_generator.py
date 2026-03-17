"""
Here we have an object class: bddl generator.
Each bddl generator generates for a specific type of subtask.

This is AI refined version with comments and simplified parameters, though there maybe bugs which we need to test.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .variables import _PROBLEM_CLASS, OBJECT_NUM_LIMITS, OBJECT_POOL, BOWL_TYPE, INSTRUCTION_TEMPLATES, OBJECT_SIZE_RANK, BOWL_SIZE_RANK, BOWL_POOL, debug_log_print
from .env_generate_utils import allocate_obj_to_region, parse_cell_region, parse_ranking_index, clamp_rank_to_list

import os
import re
import random
import tempfile
import textwrap
import numpy as np

def generate_bddl(
        resolved_language: str,
        inst2region,
        regions,
        all_objects,
        target_object: str,
        bowl_instance: str = f"{BOWL_TYPE}_0",
        save_bddl: bool = True,
        output_path: str = None
):
    """
    Assembles and optionally writes a BDDL problem file from pre-resolved scene components.
 
    This is the shared low-level writer called by all task-specific generators. It takes
    a fully resolved instruction string, a mapping from object instances to grid regions,
    the region coordinate dictionary, the full object list, and the identities of the
    target object and destination bowl, then formats everything into a valid BDDL file.
 
    Returns a dict containing the raw BDDL string, the output path, the target object,
    the destination bowl, the target's assigned region, and the resolved instruction.
    """
 
    # ── :regions block ────────────────────────────────────────────────────────
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
    type_to_insts: dict = {}
    for inst in all_objects:
        obj_type = inst.rsplit("_", 1)[0]
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
        "bddl_path":         os.path.abspath(output_path) if output_path else None,
        "target_object":     target_object,
        "target_place":      bowl_instance,
        "target_region":     inst2region[target_object],
        "resolved_language": resolved_language
    }
 
 

# ─────────────────────────────────────────────────────────────────────────────
def generate_egocentric_pick_task_bddl(
    language: str,
    object_type: str=None,
    bow_type: str= None,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a pick task using an egocentric (camera-relative) depth-rank predicate.
 
    All objects are of the same type, sampled from OBJECT_POOL. They are sorted by
    row index (camera depth), and the target is selected by parsing the ordinal rank
    from the language instruction (e.g. "closest", "2nd furthest"). The generic token
    "object" or "item" in the instruction is replaced with the sampled object type.
    """
    rng = np.random.RandomState(seed)
 
    # pick object only if its not provided
    if object_type is None:
        object_type = rng.choice(OBJECT_POOL) 


    if bow_type is None:
        bow_type = rng.choice(BOWL_POOL) 

    resolved_language = language.replace("object", object_type).replace("item", object_type)
 
    obj_num = min(OBJECT_NUM_LIMITS[object_type], obj_num)
    obj_list = [f"{object_type}_{i}" for i in range(obj_num)]
 
    inst2region, regions = allocate_obj_to_region(
        obj_list,
        has_bowl=True,
        grid_size=grid_size,
        seed=seed,
        need_middle_object=False,
        bowl_type=bow_type
    )
 
    sorted_objects = sorted(
        [obj for obj in inst2region if bow_type not in obj],
        key=lambda obj: parse_cell_region(inst2region[obj])[0],
        reverse=True
    )
 
    rank = parse_ranking_index(resolved_language, obj_num)
    rank, resolved_language = clamp_rank_to_list(rank, resolved_language, sorted_objects, rng)

    debug_log_print(function_name="generate_egocentric_pick_task_bddl", debug_message=f"instruction '{resolved_language}': rank={rank}", is_debugging=is_debugging)

    target_object = sorted_objects[rank]

    debug_log_print(function_name="generate_egocentric_pick_task_bddl", debug_message=f"target={target_object}", is_debugging=is_debugging)
 
    return generate_bddl(
        resolved_language=resolved_language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_object,
        bowl_instance=f"{bow_type}_0",
        save_bddl=save_bddl,
        output_path=output_path
    )
 
 
def generate_egocentric_place_task_bddl(
    language: str,
    object_type: str = None,
    bow_type: str = None,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a place task using an egocentric (camera-relative) depth-rank predicate
    over bowls.
 
    A single object is sampled from OBJECT_POOL as the pick target. Multiple bowls are
    placed on the table, sorted by row index, and the destination bowl is selected by
    parsing the ordinal rank from the language instruction (e.g. "place in the closest
    bowl"). The generic token "object"/"item" in the instruction is replaced with the
    sampled object type.
    """
    rng = np.random.RandomState(seed)

    if object_type is None:
        object_type = rng.choice(OBJECT_POOL)

    if bow_type is None:
        bow_type = rng.choice(BOWL_POOL)
 
    obj_num = min(OBJECT_NUM_LIMITS[bow_type], obj_num)
 
    target_pick_obj = f"{object_type}_0"
    resolved_language = language.replace("object", object_type).replace("item", object_type)
 
    obj_list = [target_pick_obj] + [f"{bow_type}_{i}" for i in range(obj_num)]
 
    inst2region, regions = allocate_obj_to_region(
        obj_list,
        has_bowl=False,
        grid_size=grid_size,
        seed=seed,
        need_middle_object=False,
        bowl_type=bow_type
    )
 
    sorted_bowls = sorted(
        [obj for obj in inst2region if bow_type in obj],
        key=lambda obj: parse_cell_region(inst2region[obj])[0],
        reverse=True
    )
 
    rank = parse_ranking_index(resolved_language, obj_num)
    rank, resolved_language = clamp_rank_to_list(rank, resolved_language, sorted_bowls, rng)

    debug_log_print(function_name="generate_egocentric_place_task_bddl", debug_message=f"instruction '{resolved_language}': rank={rank}", is_debugging=is_debugging)

    target_bowl_instance = sorted_bowls[rank]

    debug_log_print(function_name="generate_egocentric_place_task_bddl", debug_message=f"target={target_bowl_instance}", is_debugging=is_debugging)
    
 
    return generate_bddl(
        resolved_language=resolved_language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_pick_obj,
        bowl_instance=target_bowl_instance,
        save_bddl=save_bddl,
        output_path=output_path
    )
 
 
def generate_allocentric_pick_task_bddl(
    language: str,
    object_type: str = None,
    bow_type: str = None,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a pick task using an allocentric depth-rank predicate relative to a bowl.
 
    All objects are of the same type and are sorted by their Euclidean distance to
    bowl_0. The allocation guarantees all inter-object distances to the bowl are unique.
    The target is selected by parsing the ordinal rank from the language instruction
    (e.g. "pick the object closest to the bowl"). The generic token "object"/"item"
    is replaced with the sampled object type.
    """
    rng = np.random.RandomState(seed)

    if object_type is None:
        object_type = rng.choice(OBJECT_POOL)

    if bow_type is None:
        bow_type = rng.choice(BOWL_POOL)

    obj_num = min(OBJECT_NUM_LIMITS[object_type], obj_num)
    obj_list = [f"{object_type}_{i}" for i in range(obj_num)]

    resolved_language = language.replace("object", object_type).replace("item", object_type)
 
    inst2region, regions, distance_map = allocate_obj_to_region(
        obj_list,
        has_bowl=True,
        grid_size=grid_size,
        seed=seed,
        need_middle_object=False,
        bowl_type=bow_type,
        need_allocation_dist=True,
        allocated_object_type=f"{bow_type}_0",
    )
 
    sorted_objects = sorted(
        [obj for obj in inst2region if bow_type not in obj],
        key=lambda obj: distance_map[obj]
    )
 
    rank = parse_ranking_index(resolved_language, obj_num)
    rank, resolved_language = clamp_rank_to_list(rank, resolved_language, sorted_objects, rng)

    debug_log_print(function_name="generate_allocentric_pick_task_bddl", debug_message=f"instruction '{resolved_language}': rank={rank}", is_debugging=is_debugging)

    target_object = sorted_objects[rank]

    debug_log_print(function_name="generate_allocentric_pick_task_bddl", debug_message=f"target={target_object}", is_debugging=is_debugging)
 
    return generate_bddl(
        resolved_language=resolved_language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_object,
        save_bddl=save_bddl,
        output_path=output_path
    )
 
 
def generate_allocentric_place_task_bddl(
    language: str,
    object_type: str = None,
    bow_type: str = None,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 10,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a place task using an allocentric depth-rank predicate relative to the
    pick object.
 
    A single object is sampled as the pick target. Multiple bowls are placed on the
    table and sorted by their Euclidean distance to the pick object. The destination
    bowl is selected by parsing the ordinal rank from the language instruction (e.g.
    "place in the bowl closest to it"). The allocation guarantees all bowl-to-object
    distances are unique.
    """
    rng = np.random.RandomState(seed)

    if object_type is None:
        object_type = rng.choice(OBJECT_POOL)

    if bow_type is None:
        bow_type = rng.choice(BOWL_POOL)

    obj_num = min(OBJECT_NUM_LIMITS[bow_type], obj_num)
 
    target_pick_obj = f"{object_type}_0"
    resolved_language = language.replace("object", object_type).replace("item", object_type)
 
    obj_list = [target_pick_obj] + [f"{bow_type}_{i}" for i in range(obj_num)]
 
    inst2region, regions, distance_map = allocate_obj_to_region(
        obj_list,
        has_bowl=False,
        grid_size=grid_size,
        seed=seed,
        need_middle_object=False,
        bowl_type=bow_type,
        need_allocation_dist=True,
        allocated_object_type=target_pick_obj,
    )
 
    sorted_bowls = sorted(
        [obj for obj in inst2region if bow_type in obj],
        key=lambda obj: distance_map[obj]
    )
 
    rank = parse_ranking_index(resolved_language, len(sorted_bowls))
    rank, resolved_language = clamp_rank_to_list(rank, resolved_language, sorted_bowls, rng)

    debug_log_print(function_name="generate_allocentric_place_task_bddl", debug_message=f"instruction '{resolved_language}': rank={rank}", is_debugging=is_debugging)

    target_bowl_instance = sorted_bowls[rank]

    debug_log_print(function_name="generate_allocentric_place_task_bddl", debug_message=f"target={target_bowl_instance}", is_debugging=is_debugging)
 
    return generate_bddl(
        resolved_language=resolved_language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_pick_obj,
        bowl_instance=target_bowl_instance,
        save_bddl=save_bddl,
        output_path=output_path,
    )
 
 
 

def generate_middle_pick_task_bddl(
    language: str,
    bow_type: str = None,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a pick task where the target is the depth-median object on the table.
 
    Randomly places 3, 5, or 7 heterogeneous objects sampled from OBJECT_POOL.
    Objects are sorted by their row index (a proxy for camera depth), and the
    median-ranked object is selected as the pick target. The language instruction
    is expected to contain a "middle" predicate.
    """
    rng = np.random.RandomState(seed)

    if bow_type is None:
        bow_type = rng.choice(BOWL_POOL)
 
    obj_num = rng.choice([3, 5, 7])
 
    obj_list = []
    obj_count_dict = {}
    for _ in range(obj_num):
        cur_obj_type = rng.choice(OBJECT_POOL)
        if cur_obj_type in obj_count_dict:
            obj_count_dict[cur_obj_type] += 1
        else:
            obj_count_dict[cur_obj_type] = 0
        obj_list.append(f"{cur_obj_type}_{obj_count_dict[cur_obj_type]}")
 
    inst2region, regions = allocate_obj_to_region(
        obj_list,
        has_bowl=True,
        grid_size=grid_size,
        seed=seed,
        need_middle_object=True,
        bowl_type=bow_type
    )
 
    sorted_objects = sorted(
        [obj for obj in inst2region if bow_type not in obj],
        key=lambda obj: parse_cell_region(inst2region[obj])[0],
        reverse=True
    )
 
    debug_log_print(function_name="generate_middle_pick_task_bddl", debug_message=f"sorted objects: {sorted_objects}", is_debugging=is_debugging)

    middle_object = sorted_objects[len(sorted_objects) // 2]

    debug_log_print(function_name="generate_middle_pick_task_bddl", debug_message=f"target={middle_object}", is_debugging=is_debugging)
 
    return generate_bddl(
        resolved_language=language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=middle_object,
        bowl_instance=f"{bow_type}_0",
        save_bddl=save_bddl,
        output_path=output_path
    )
 
 
def generate_middle_place_task_bddl(
    language: str,
    object_type: str = None,
    bow_type: str = None,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a place task where the destination is the depth-median bowl on the table.
 
    Randomly places 3, 5, or 7 bowls alongside a single randomly sampled pick object.
    Bowls are sorted by row index, and the median-ranked bowl is selected as the
    placement destination. The language instruction is expected to contain a "middle"
    predicate referencing the bowl.
    """
    rng = np.random.RandomState(seed)

    if object_type is None:
        object_type = rng.choice(OBJECT_POOL)

    if bow_type is None:
        bow_type = rng.choice(BOWL_POOL)
 
    bowl_num = rng.choice([3, 5, 7])
 
    target_obj_type = f"{object_type}_0"
    obj_list = [target_obj_type]
    for i in range(bowl_num):
        obj_list.append(f"{bow_type}_{i}")
 
    inst2region, regions = allocate_obj_to_region(
        obj_list,
        has_bowl=False,
        grid_size=grid_size,
        seed=seed,
        need_middle_object=True,
    )
 
    sorted_bowls = sorted(
        [obj for obj in inst2region if bow_type in obj],
        key=lambda obj: parse_cell_region(inst2region[obj])[0],
        reverse=True
    )
 
    debug_log_print(function_name="generate_middle_place_task_bddl", debug_message=f"sorted bowls: {sorted_bowls}", is_debugging=is_debugging)

    middle_bowl = sorted_bowls[len(sorted_bowls) // 2]

    debug_log_print(function_name="generate_middle_place_task_bddl", debug_message=f"target={middle_bowl}", is_debugging=is_debugging)
 
    return generate_bddl(
        resolved_language=language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_obj_type,
        bowl_instance=middle_bowl,
        save_bddl=save_bddl,
        output_path=output_path
    )
 
 

def generate_pick_by_feature_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 5,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    target_object_type: str = None,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a pick task where the target is selected by a size-rank predicate
    (e.g. "pick the largest object", "pick the smallest object").
 
    Objects are drawn without replacement from OBJECT_SIZE_RANK, which is a list of
    object types ordered by ascending physical size. The target is identified by
    parsing the ordinal rank from the language instruction and indexing into the
    size-sorted object list.
    """
    rng = np.random.RandomState(seed)
 
    obj_num = min(len(OBJECT_SIZE_RANK), obj_num)
    obj_list = [f"{t}_0" for t in rng.choice(OBJECT_SIZE_RANK, size=obj_num, replace=False)]
 
    inst2region, regions = allocate_obj_to_region(
        obj_list,
        has_bowl=True,
        grid_size=grid_size,
        seed=seed,
        need_middle_object=False,
        bowl_type=BOWL_TYPE
    )
 
    sorted_objects = sorted(
        [inst for inst in inst2region if BOWL_TYPE not in inst],
        key=lambda inst: OBJECT_SIZE_RANK.index(inst.rsplit("_", 1)[0])
    )
 
    rank = parse_ranking_index(language, obj_num)
    rank, language = clamp_rank_to_list(rank, language, sorted_objects, rng)

    debug_log_print(function_name="generate_pick_by_feature_task_bddl", debug_message=f"instruction '{language}': rank={rank}", is_debugging=is_debugging)

    target_object = sorted_objects[rank]

    debug_log_print(function_name="generate_pick_by_feature_task_bddl", debug_message=f"target={target_object}", is_debugging=is_debugging)
 
    return generate_bddl(
        resolved_language=language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_object,
        save_bddl=save_bddl,
        output_path=output_path
    )
 
 
def generate_place_by_feature_task_bddl(
    language: str,
    seed: Optional[int] = None,
    grid_size: Optional[int] = 20,
    obj_num: int = 5,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False,
) -> dict:
    """
    Generates a place task where the destination bowl is selected by a size-rank
    predicate (e.g. "place in the largest bowl", "place in the smallest bowl").
 
    Bowls are drawn without replacement from BOWL_SIZE_RANK, which is a list of bowl
    types ordered by ascending physical size. A single pick object is sampled from
    OBJECT_POOL. The destination bowl is identified by parsing the ordinal rank from
    the language instruction and indexing into the size-sorted bowl list.
    """
    rng = np.random.RandomState(seed)
 
    bowl_num = min(len(BOWL_SIZE_RANK), obj_num)
    obj_list = [f"{t}_0" for t in rng.choice(BOWL_SIZE_RANK, size=bowl_num, replace=False)]
 
    target_obj_type = f"{rng.choice(OBJECT_POOL)}_0"
    obj_list.append(target_obj_type)
 
    inst2region, regions = allocate_obj_to_region(
        obj_list,
        has_bowl=False,
        grid_size=grid_size,
        seed=seed
    )
 
    sorted_bowls = sorted(
        [inst for inst in inst2region if target_obj_type not in inst],
        key=lambda inst: BOWL_SIZE_RANK.index(inst.rsplit("_", 1)[0])
    )
 
    debug_log_print(function_name="generate_place_by_feature_task_bddl", debug_message=f"sorted bowls: {sorted_bowls}", is_debugging=is_debugging)
 
    rank = parse_ranking_index(language, obj_num)
    rank, language = clamp_rank_to_list(rank, language, sorted_bowls, rng)

    debug_log_print(function_name="generate_place_by_feature_task_bddl", debug_message=f"instruction '{language}': rank={rank}", is_debugging=is_debugging)

    target_bowl = sorted_bowls[rank]

    debug_log_print(function_name="generate_place_by_feature_task_bddl", debug_message=f"target={target_bowl}", is_debugging=is_debugging)
 
    return generate_bddl(
        resolved_language=language,
        inst2region=inst2region,
        regions=regions,
        all_objects=obj_list,
        target_object=target_obj_type,
        bowl_instance=target_bowl,
        save_bddl=save_bddl,
        output_path=output_path
    )
 
 
def generate_random_rank_task_bddl(
    language: str = random.choice(INSTRUCTION_TEMPLATES),
    object_type: str=None,
    bow_type: str=None,
    seed: Optional[int] = None,
    num_objects: int = 10,
    grid_size: Optional[int] = 20,
    output_path: Optional[str] = None,
    save_bddl: bool = False,
    is_debugging: bool = False
) -> dict:
    """
    Dispatches a language instruction to the appropriate task-specific BDDL generator.
 
    Parses the instruction string using a priority-ordered set of regex rules to
    identify which task type the instruction describes, then delegates to the
    corresponding generator function. Covers all eight task categories: middle pick,
    middle place, feature pick, feature place, allocentric pick, allocentric place,
    egocentric pick, and egocentric place.
    """
    lang = language.lower().strip()
 
    # ── middle place: "place in the bowl in the middle" ──────────────────────
    if "middle" in lang and re.search(r"place.*bowl.*middle|bowl.*middle", lang):
        debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → middle place", is_debugging=is_debugging)
        return generate_middle_place_task_bddl(
            language=language, object_type=object_type, bow_type=bow_type, seed=seed, grid_size=grid_size,
            output_path=output_path, save_bddl=save_bddl, is_debugging=is_debugging
        )
 
    # ── middle pick: "pick the object in the middle" ─────────────────────────
    if "middle" in lang:
        debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → middle pick", is_debugging=is_debugging)
        return generate_middle_pick_task_bddl(
            language=language, bow_type=bow_type, seed=seed, grid_size=grid_size,
            output_path=output_path, save_bddl=save_bddl, is_debugging=is_debugging
        )
 
    # ── feature pick/place (largest / smallest) ───────────────────────────────
    if re.search(r"\b(largest|smallest)\b", lang):
        if re.search(r"\bbowl\b", lang):
            debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → feature place", is_debugging=is_debugging)
            return generate_place_by_feature_task_bddl(
                language=language, seed=seed, grid_size=grid_size,
                obj_num=num_objects, output_path=output_path, save_bddl=save_bddl, is_debugging=is_debugging
            )
        else:
            debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → feature pick", is_debugging=is_debugging)
            return generate_pick_by_feature_task_bddl(
                language=language, seed=seed, grid_size=grid_size,
                obj_num=num_objects, output_path=output_path, save_bddl=save_bddl, is_debugging=is_debugging
            )
 
    # ── allocentric place: "bowl closest/furthest to it" ─────────────────────
    if re.search(r"bowl.*(closest|furthest|furtherest|farthest).*\bit\b", lang):
        debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → allocentric place", is_debugging=is_debugging)
        return generate_allocentric_place_task_bddl(
            language=language, object_type=object_type, bow_type=bow_type, seed=seed, grid_size=grid_size,
            obj_num=num_objects, output_path=output_path, save_bddl=save_bddl, is_debugging=is_debugging
        )
 
    # ── allocentric pick: "object closest/furthest to the bowl" ──────────────
    if re.search(r"pick the (?:\d+(?:st|nd|rd|th)?\s+)?object\s+(?:closest|furthest|furtherest|farthest)\s+to the bowl", lang):
        debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → allocentric pick", is_debugging=is_debugging)
        return generate_allocentric_pick_task_bddl(
            language=language, object_type=object_type, bow_type=bow_type, seed=seed, grid_size=grid_size,
            obj_num=num_objects, output_path=output_path, save_bddl=save_bddl, is_debugging=is_debugging
        )
 
    # ── egocentric place: "place in the [rank] bowl" ─────────────────────────
    if re.search(r"place in the (?:\d+(?:st|nd|rd|th)?\s+)?(?:closest|furthest|furtherest|farthest)\s+bowl\s*\.", lang):
        debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → egocentric place", is_debugging=is_debugging)
        return generate_egocentric_place_task_bddl(
            language=language, object_type=object_type, bow_type=bow_type, seed=seed, grid_size=grid_size,
            obj_num=num_objects, output_path=output_path, save_bddl=save_bddl, is_debugging=is_debugging
        )
 
    # ── egocentric pick: "pick the [rank] object" ────────────────────────────
    if re.search(r"pick the.*(closest|furthest|furtherest|farthest|\d+\w*)\s+object", lang):
        debug_log_print(function_name="generate_random_rank_task_bddl", debug_message=f"'{language}' → egocentric pick", is_debugging=is_debugging)
        return generate_egocentric_pick_task_bddl(
            language=language, object_type=object_type, seed=seed, grid_size=grid_size,
            obj_num=num_objects, output_path=output_path, bow_type=bow_type, save_bddl=save_bddl, is_debugging=is_debugging
        )