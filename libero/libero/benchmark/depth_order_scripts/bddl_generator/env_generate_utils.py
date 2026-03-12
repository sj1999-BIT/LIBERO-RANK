"""
What we do here is to create helper functions for the libero rank generation.

Total 4 types of environment

Type 1: N same objects + 1 bowl
Type 2: N bowls + 1 object
Type 3: 7 different objects + 1 bowl

"""

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
                           need_middle_object: bool = False
                           ):
    
    # Create a random number generator with the given seed for reproducibility
    rng = np.random.RandomState(seed) 

    # create the regions
    regions = {**make_table_regions(grid_size), **(extra_regions or {})}


    # add a bowl instance
    if has_bowl:
        obj_list += [f"{BOWL_TYPE}_0"]
    
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

            if need_middle_object:
                # success check: one object must be the median on both x and y axes
                non_bowl = [obj for obj in inst2region if BOWL_TYPE not in obj]
                sorted_by_x = sorted(non_bowl, key=lambda o: parse_cell_region(inst2region[o])[0])
                sorted_by_y = sorted(non_bowl, key=lambda o: parse_cell_region(inst2region[o])[1])
                mid = len(non_bowl) // 2
                if sorted_by_x[mid] != sorted_by_y[mid]:
                    continue  # no single object is the center on both axes, retry

                print(f"[DEBUG INFO allocate_obj_to_region]: found mid object {sorted_by_x[mid]}")

                
            # success
            return inst2region, regions

        except (ValueError):
            continue  # retry with new rng choices

            
    raise RuntimeError(
        f"Could not place all {len(obj_list)} objects after "
        f"{max_attempt} attempts. Try reducing num_objects, spacing, or increasing grid_size."
    )




