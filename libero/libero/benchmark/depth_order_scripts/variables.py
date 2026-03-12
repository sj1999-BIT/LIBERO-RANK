
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

