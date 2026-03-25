"""
test_object_spacing.py
────────────────────────────────────────────────────────────────────────────────
Dense-packing spacing calibration tool for LIBERO_DEPTH_ORDER tasks.

Instead of the normal allocate_obj_to_region (which removes the entire row on
placement and enforces one-object-per-depth-level), this script uses a greedy
row-first packer:

    • Walk grid cells in row-major order (row 0 → N, col 0 → M within each row).
    • Place an object in the first cell whose Chebyshev neighbourhood is clear.
    • Do NOT clear the whole row — keep filling the same row until no cell fits,
      then move to the next row.

This produces the densest possible layout for a given object type / spacing,
making it easy to eyeball whether the spacing constant is too tight or too loose
in simulation.

Usage
─────
    # random object from the combined pool
    python test_object_spacing.py

    # specific object, default grid
    python test_object_spacing.py --object milk

    # bowl type, bigger grid, more objects
    python test_object_spacing.py --object akita_black_bowl --grid 25 --num 30

    # save the generated BDDL
    python test_object_spacing.py --object butter --save --out /tmp/spacing_test.bddl

    # quiet: no ASCII map, just the BDDL
    python test_object_spacing.py --object plate --quiet
"""

import argparse
import os
import random
import sys
import textwrap
import tempfile

import numpy as np

# ── inline copies of the constants we need ───────────────────────────────────
# (so the script works without the full project on PYTHONPATH)

from libero.libero.benchmark.rank_scripts.bddl_generator import (
    OBJECT_POOL,
    BOWL_POOL,
    OBJECT_SPACING_REQUIREMENTS,
)

_PROBLEM_CLASS = "LIBERO_TABLETOP_MANIPULATION"
COMBINED_POOL = OBJECT_POOL + BOWL_POOL

# ── grid helpers (mirrors env_generate_utils) ─────────────────────────────────

X_RANGE = 0.2
Y_RANGE = 0.25

def make_table_regions(n: int, lower_limit: int=None, upper_limit: int=None) -> dict:
    """Generate an n×n grid; rows 0-3 and (n-4)-(n-1) are trimmed (reachability)."""
    regions = {}
    xs = np.linspace(-X_RANGE, X_RANGE, n + 1)
    ys = np.linspace(-Y_RANGE, Y_RANGE, n + 1)
    for i, (x0, x1) in enumerate(zip(xs[:-1], xs[1:])):
        if lower_limit is not None and i < lower_limit:
            continue

        if upper_limit is not None and i > upper_limit:
            continue

        for j, (y0, y1) in enumerate(zip(ys[:-1], ys[1:])):
            if lower_limit is not None and j < lower_limit:
                continue

            if upper_limit is not None and j > upper_limit:
                continue
            regions[f"cell_{i}_{j}"] = (float(x0), float(y0), float(x1), float(y1))
    return regions


def parse_cell(name: str):
    _, i, j = name.split("_")
    return int(i), int(j)


# ── dense packer ─────────────────────────────────────────────────────────────

def pack_objects_dense(
    obj_type: str,
    max_objects: int,
    grid_size: int = 20,
    seed: int = None,
):
    """
    Greedy row-first packer.

    Fills each row completely before moving to the next.  Only removes cells
    within the Chebyshev spacing neighbourhood of each placed object — never
    the full row — so multiple objects can share the same depth level.

    Returns
    -------
    inst2region : dict  { "obj_type_N": "cell_i_j", … }
    regions     : dict  { "cell_i_j": (x0, y0, x1, y1), … }
    """
    spacing = OBJECT_SPACING_REQUIREMENTS.get(obj_type, 4)
    regions = make_table_regions(grid_size)

    # Sort cells row-major (ascending i, then ascending j within each row)
    all_cells = sorted(regions.keys(), key=lambda c: (parse_cell(c)[0], parse_cell(c)[1]))

    occupied: set[str] = set()   # cells blocked by a prior placement
    inst2region: dict = {}
    obj_idx = 0

    for cell in all_cells:
        if obj_idx >= max_objects:
            break
        if cell in occupied:
            continue

        # Place object here
        inst_name = f"{obj_type}_{obj_idx}"
        inst2region[inst_name] = cell
        obj_idx += 1

        # Block the Chebyshev neighbourhood (but NOT the whole row/col)
        ri, rj = parse_cell(cell)
        for c in all_cells:
            ci, cj = parse_cell(c)
            if max(abs(ci - ri), abs(cj - rj) * Y_RANGE / X_RANGE) <= spacing:
                occupied.add(c)
            if (ci == ri):
                occupied.add(c)



    return inst2region, regions


# ── ASCII visualiser ──────────────────────────────────────────────────────────

def ascii_grid(inst2region: dict, regions: dict, obj_type: str) -> str:
    """
    Render the placement as an ASCII grid.

    Rows increase bottom-to-top (high-i = far from camera = top of table).
    Columns increase left-to-right.
    'O' = object placed, '.' = empty usable cell.
    """
    if not regions:
        return "(no regions)"

    all_rows = sorted({parse_cell(c)[0] for c in regions})
    all_cols = sorted({parse_cell(c)[1] for c in regions})

    placed = set(inst2region.values())

    # header
    col_header = "     " + "".join(f"{j:2d}" for j in all_cols)
    lines = [col_header, "     " + "--" * len(all_cols)]

    for i in reversed(all_rows):          # far rows at top
        row_str = f"r{i:2d} |"
        for j in all_cols:
            cell = f"cell_{i}_{j}"
            if cell not in regions:
                row_str += "  "
            elif cell in placed:
                row_str += " O"
            else:
                row_str += " ."
        lines.append(row_str)

    lines.append("")
    lines.append(f"  Camera / robot base  (row 0 = closest to camera)")
    lines.append(f"  O = {obj_type}   spacing={OBJECT_SPACING_REQUIREMENTS.get(obj_type,'?')}")
    return "\n".join(lines)


# ── BDDL writer ───────────────────────────────────────────────────────────────

def write_bddl(
    inst2region: dict,
    regions: dict,
    obj_type: str,
    output_path: str = None,
    save: bool = False,
) -> str:
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

    obj_names = sorted(inst2region.keys())
    objects_line = f"    {' '.join(obj_names)} - {obj_type}"

    init_lines = [
        f"    (On {inst} main_table_{rname})"
        for inst, rname in inst2region.items()
    ]

    NL = "\n"
    bddl = textwrap.dedent(f"""\
        (define (problem {_PROBLEM_CLASS})
          (:domain robosuite)
          (:language [SPACING TEST] Pick any {obj_type} and place in bowl.)
          (:regions
        {NL.join(region_lines)}
          )
          (:fixtures
            main_table - table
          )
          (:objects
        {objects_line}
          )
          (:obj_of_interest
            {obj_names[0]}
          )
          (:init
        {NL.join(init_lines)}
          )
          (:goal
            (And (On {obj_names[0]} main_table_cell_0_0))
          )
        )
    """)

    if save:
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".bddl", prefix="spacing_test_")
            os.close(fd)
        with open(output_path, "w") as f:
            f.write(bddl)
        print(f"\n[BDDL saved → {os.path.abspath(output_path)}]")

    return bddl, output_path


# ── CLI ───────────────────────────────────────────────────────────────────────

import numpy as np
import cv2
import math

def combine_images_to_grid(images: list, output_path: str) -> np.ndarray:
    """
    Combines a list of n*n images into a single square grid image and saves it.

    Args:
        images      : list of (H, W, C) uint8 arrays, length must be a perfect square
        output_path : path to save the combined image (e.g. "/tmp/grid.png")

    Returns:
        The combined grid as a (n*H, n*W, C) uint8 array.
    """
    n = int(math.isqrt(len(images)))
    assert n * n == len(images), \
        f"Expected a perfect-square number of images, got {len(images)}"

    h, w, c = images[0].shape
    assert all(img.shape == (h, w, c) for img in images), \
        "All images must have the same shape"

    rows = [np.concatenate(images[i*n : (i+1)*n], axis=1) for i in range(n)]
    grid = np.concatenate(rows, axis=0)

    cv2.imwrite(output_path, grid)
    return grid

def main():
    import cv2
    import random
    from tqdm import tqdm
    from libero.libero.envs import OffScreenRenderEnv
    grid=20
    num=100
    env_grid_len = 1
    
    save=False
    out="/root/code/LIBERO-RANK/testing_results"
    if not os.path.exists(out):
        os.makedirs(out)

    img_dir = os.path.join(out, "images")

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)


    for obj_type in tqdm(COMBINED_POOL, desc="progressing each object"):
        imgs = []
        for _ in range(env_grid_len * env_grid_len):
            seed = int(random.random() * 10000)
            if obj_type not in OBJECT_SPACING_REQUIREMENTS:
                print(f"ERROR: unknown object type '{obj_type}'.  Run with --list to see valid types.")
                sys.exit(1)

            spacing = OBJECT_SPACING_REQUIREMENTS[obj_type]

            # pack
            inst2region, regions = pack_objects_dense(
                obj_type=obj_type,
                max_objects=num,
                grid_size=grid,
                seed=seed,
            )

            placed_count = len(inst2region)
            total_cells  = len(regions)


            # print(f"\n{'─'*60}")
            # print(f"  Object type : {obj_type}")
            # print(f"  Spacing req : {spacing} cells (Chebyshev)")
            # print(f"  Grid size   : {grid}×{grid}  ({total_cells} usable cells)")
            # print(f"  Placed      : {placed_count} objects")
            # print(f"  Density     : {placed_count/total_cells*100:.1f}% of usable cells occupied")
            # print(f"{'─'*60}\n")
            # print(ascii_grid(inst2region, regions, obj_type))
            # print()

            save = save or (out is not None)
            bddl, bddl_output_path = write_bddl(
                inst2region=inst2region,
                regions=regions,
                obj_type=obj_type,
                output_path=os.path.join(out, f"{obj_type}.bddl"),
                save=save,
            )

            # # pretty-print the objects block so you can see all placements at a glance
            # print("\n── Placement summary ──────────────────────────────────────")
            # for inst, region in sorted(inst2region.items()):
            #     ri, rj = parse_cell(region)
            #     x0, y0, x1, y1 = regions[region]
            #     cx, cy = (x0+x1)/2, (y0+y1)/2
            #     print(f"  {inst:<35s}  row={ri:2d} col={rj:2d}  world=({cx:+.3f}, {cy:+.3f})")
            # print()

            if placed_count == 0:
                print("WARNING: no objects could be placed — spacing may be too large for this grid size.")
                print(f"         Try --grid {grid + 5} or reduce OBJECT_SPACING_REQUIREMENTS['{obj_type}']")


            env = OffScreenRenderEnv(
                bddl_file_name=bddl_output_path,
                robots=["Panda"],
                camera_heights=256,
                camera_widths=256,    
                use_camera_obs=True,
                camera_names=["robot0_eye_in_hand", "agentview"],
                camera_depths=True,   # ← add
            )


            env.seed(seed)
            obs = env.reset()

            img = obs["agentview_image"]

            rgb = img[..., ::-1]
            rgb = np.flipud(rgb).copy()

            imgs.append(rgb)
            
            
        
        combine_images_to_grid(imgs, os.path.join(img_dir, f"{obj_type}_dense.jpg"))

        env.close()


if __name__ == "__main__":
    main()