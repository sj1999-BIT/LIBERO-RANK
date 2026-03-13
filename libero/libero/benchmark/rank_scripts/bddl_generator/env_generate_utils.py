"""
What we do here is to create helper functions for the libero rank generation.

Total 4 types of environment

Type 1: N same objects + 1 bowl
Type 2: N bowls + 1 object
Type 3: 7 different objects + 1 bowl

"""

import re
import numpy as np


##############################################################################################
# Table region helper functions to allocate the correct regions for each object
##############################################################################################

"""
Generate an n×n grid of table regions.
e.g. n=4 → 16 cells, n=5 → 25 cells.
"""
def make_table_regions(n: int) -> dict:
    regions = {}
    xs = np.linspace(-0.20, 0.20, n + 1)
    ys = np.linspace(-0.25, 0.25, n + 1)
    for i, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        for j, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            regions[f"cell_{i}_{j}"] = (float(x0), float(y0), float(x1), float(y1))
    return regions


"""
# region_name is in the format "cell_i_j"
This function mostly used to get the position of each region so as to get the rank of each object.
"""
def parse_cell_region(region_name):
    _, i, j = region_name.split("_")
    return int(i), int(j)


"""
Remove all regions in the same row as the occupied region (enforcing one object per row),
plus all regions within `spacing` cells in any direction around the occupied region.

spacing=1 → removes immediate neighbours (original behaviour)
spacing=2 → removes 2-cell radius around occupied cell
spacing=0 → removes only the same row (no extra buffer)
"""
def remove_regions(region_names, occupied_region_name, spacing: int = 1, same_col_allowed: bool = True):

    occupied_row, occupied_col = parse_cell_region(occupied_region_name)
    
    # Remove entire row (enforces one object per x-depth level)
    to_remove = [r for r in region_names if parse_cell_region(r)[0] == occupied_row]
    for r in to_remove:
        if r in region_names:
            region_names.remove(r)

    # Remove entire col if required (enforces one object per x-depth level)
    if not same_col_allowed:
        to_remove = [r for r in region_names if parse_cell_region(r)[1] == occupied_col]
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

    return region_names


#########################################################################
# Allocate each object to appropriate region
#########################################################################

from .variables import OBJECT_SPACING_REQUIREMENTS, BOWL_TYPE
from typing import Optional


"""
Allcate region for the objects in the list.
assume a bowl is needed to be present in the scene as well.
Obj_list is a list of strings, each item is objectype_index
returns 2 dicts: inst2region, regions
1. inst2region: each obj_type_index mapped to the region name
2. regions: each region name mapped to the actual float value
"""
def allocate_obj_to_region(obj_list, 
                           has_bowl=True, 
                           grid_size: Optional[int] = 20,
                           extra_regions: Optional[dict] = None,
                           max_attempt: Optional[int] = 100,
                           seed: Optional[int] = None,
                           need_middle_object: bool = False,
                           need_allocation_dist: bool = False,
                           allocated_object_type: str = None,
                           bowl_type: str = BOWL_TYPE
                           ):
    
    # Create a random number generator with the given seed for reproducibility
    rng = np.random.RandomState(seed) 

    # create the regions
    regions = {**make_table_regions(grid_size), **(extra_regions or {})}


    # add a bowl instance
    if has_bowl:
        obj_list += [f"{bowl_type}_0"]
    
    for _ in range(max_attempt):
        # get a list of all the region names for easier removal
        region_names = list(regions.keys())

        # final result is a dict with each object type refering to a region
        inst2region = {}

        try:
            # assign each item to a unique row with spacing
            for obj in obj_list:
                if not region_names:
                    raise ValueError("No regions left for objects.")
                
                # get the type by removing the instance index (e.g. "plate_0" → "plate")
                obj_type = obj.rsplit("_", 1)[0]
                inst2region[obj] = rng.choice(region_names)
                # remove regions to prevent overlap, each object has a its own proximit to be removed
                remove_regions(region_names, inst2region[obj], spacing=OBJECT_SPACING_REQUIREMENTS[obj_type])

            # need to make sure one object is centered
            if need_middle_object:
                # success check: one object must be the median on both x and y axes
                non_bowl = [obj for obj in inst2region if BOWL_TYPE not in obj]
                sorted_by_x = sorted(non_bowl, key=lambda o: parse_cell_region(inst2region[o])[0])
                sorted_by_y = sorted(non_bowl, key=lambda o: parse_cell_region(inst2region[o])[1])
                mid = len(non_bowl) // 2
                if sorted_by_x[mid] != sorted_by_y[mid]:
                    continue  # no single object is the center on both axes, retry

                print(f"[DEBUG INFO allocate_obj_to_region]: found mid object {sorted_by_x[mid]}")

            # need to make sure all object has different distance to the target object
            if need_allocation_dist:
                if allocated_object_type is None:
                    raise RuntimeError("allocated object type cannot be None if need_allocation_dist is true")

                surrounding_obj_list = [obj for obj in obj_list if obj is not allocated_object_type]

                ref_x, ref_y = parse_cell_region(inst2region[allocated_object_type])

                # collect distance from each object to the target object
                distances = []
                for obj in surrounding_obj_list:
                    ox, oy = parse_cell_region(inst2region[obj])
                    dist_squared = ((ox - ref_x) ** 2 + (oy - ref_y) ** 2)
                    distances.append(dist_squared)

                # all distances must be unique — no two objects equidistant from reference
                if len(distances) != len(set(distances)):
                    continue  # retry allocation

                distance_map = {obj: dist for obj, dist in zip(surrounding_obj_list, distances)}

                
                return inst2region, regions, distance_map
                
            # success
            return inst2region, regions

        except (ValueError):
            continue  # retry with new rng choices

            
    raise RuntimeError(
        f"Could not place all {len(obj_list)} objects after "
        f"{max_attempt} attempts. Try reducing num_objects, spacing, or increasing grid_size."
    )


"""
get the rank that the instruction is trying to find based on the phrasing
Parse the language instruction to get a 0-based ranking index.
Ranking is by ascending x (closest = smallest x = closest to camera).
"""
def parse_ranking_index(language: str, num_items: int) -> int:

    lang = language.lower()

    # ── size-based (largest / smallest) ──────────────────────────────────────
    # list is sorted smallest → largest, so:
    #   largest  → index -1, 2nd largest → index -2, Nth largest → index -N
    #   smallest → index  0, 2nd smallest → index  1, Nth smallest → index N-1

    m = re.search(r"(\d+)(?:st|nd|rd|th)?\s+largest", lang)
    if m:
        return -int(m.group(1))

    m = re.search(r"(\d+)(?:st|nd|rd|th)?\s+smallest", lang)
    if m:
        return int(m.group(1)) - 1

    if re.search(r"\blargest\b", lang):
        return -1

    if re.search(r"\bsmallest\b", lang):
        return 0

    # ── distance-based (closest / furthest) ──────────────────────────────────

    if re.search(r"\bclosest\b", lang) and not re.search(r"\d", lang):
        return 0
    if re.search(r"\bfurthest\b|\bfurtherest\b|\bfarthest\b", lang) and not re.search(r"\d", lang):
        return -1

    m = re.search(r"(\d+)(?:st|nd|rd|th)?(?:\s+\S+)*?\s+closest", lang)
    if m:
        return int(m.group(1)) - 1

    m = re.search(r"(\d+)(?:st|nd|rd|th)?(?:\s+\S+)*?\s+(?:furthest|furtherest|farthest)", lang)
    if m:
        return -int(m.group(1))

    raise ValueError(
        f"Cannot parse ranking from language: '{language}'. "
        "Use 'closest', 'furthest', 'largest', 'smallest', or ordinals like '2nd closest', '3rd largest'."
    )
