
_PROBLEM_CLASS = "LIBERO_TABLETOP_MANIPULATION"


OBJECT_POOL = [
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

"""
Total 4 types of environment

Type 1: N same objects + 1 bowl
Type 2: N bowls + 1 object
Type 3: 3 different objects, middle must be distinguishable
Type 4: 7 different objects + 1 bowl
"""

N_SAME_OBJ_1_BOWL = 1
N_BOWL_1_OBJ = 2
THREE_DIFFERENT_OBJ = 3
ALL_DIFFERENT_OBJ = 4


