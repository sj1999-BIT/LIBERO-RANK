
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

# smallest to largest
# "ketchup", Removed cause its too similar to orange juice
# "tomato_sauce", Removed cause its too similar to alphabet_soup
# "glazed_rim_porcelain_ramekin", Removed cause its shape too different from the rest
OBJECT_SIZE_RANK = [
    "butter",
    "alphabet_soup",
    "milk",
    "orange_juice",
    "moka_pot",
    # append new object types here in size order
]

BOWL_POOL = [
    "white_bowl",
    "akita_black_bowl",
    "plate",
]

BOWL_SIZE_RANK = [
    "white_bowl",
    "akita_black_bowl",
    "plate",
    # "basket",
]

# maximum number of objects of each type to include in the task (including the bowl)
OBJECT_NUM_LIMITS = {
    "akita_black_bowl": 7,
    "milk": 7,
    "moka_pot": 7,
    "glazed_rim_porcelain_ramekin": 7,
    "tomato_sauce": 7,
    "alphabet_soup": 7,
    "butter": 7,
    "ketchup": 7,
    "orange_juice": 7,
}

# number of cells that must be between two plates (in any direction) to avoid overlap
OBJECT_SPACING_REQUIREMENTS = {
    "akita_black_bowl": 7,
    "plate": 7,
    "red_bowl": 5,
    "white_bowl": 5,
    "simpl_rack": 5,
    "basket": 5,
    "milk": 3,
    "moka_pot": 5,
    "glazed_rim_porcelain_ramekin": 4,
    "tomato_sauce": 4,
    "alphabet_soup": 3,
    "butter": 4,
    "ketchup": 3,
    "orange_juice": 3,
}



BOWL_SIZE_RANK = [
    "akita_black_bowl",
    "plate",
    "red_bowl",
    "white_bowl",
    "simpl_rack",
    "basket",
]

BOWL_TYPE = "akita_black_bowl"

"""
Total 4 types of environment

Type 1: N same objects + 1 bowl
Type 2: N bowls + 1 object
Type 3: 3 different objects, middle must be distinguishable
Type 4: 7 different objects + 1 bowl
"""


INSTRUCTION_TEMPLATES = [
    # egocentric pick tasks: many same object + 1 bowl
    "Pick the closest object and place in the bowl.",
    "Pick the furtherest object and place in the bowl.",
    "Pick the 1st closest object and place in the bowl.",
    "Pick the 2nd closest object and place in the bowl.",
    "Pick the 3rd closest object and place in the bowl.",
    "Pick the 4th closest object and place in the bowl.",
    "Pick the 5th closest object and place in the bowl.",
    "Pick the 6th closest object and place in the bowl.",
    "Pick the 7th closest object and place in the bowl.",
    "Pick the 1st furtherest object and place in the bowl.",
    "Pick the 2nd furtherest object and place in the bowl.",
    "Pick the 3rd furtherest object and place in the bowl.",
    "Pick the 4th furtherest object and place in the bowl.",
    "Pick the 5th furtherest object and place in the bowl.",
    "Pick the 6th furtherest object and place in the bowl.",
    "Pick the 7th furtherest object and place in the bowl.",
    # egocentric place tasks: 1 object + many bowls
    "Pick the object and place in the closest bowl.",
    "Pick the object and place in the furtherest bowl.",
    "Pick the object and place in the 1st closest bowl.",
    "Pick the object and place in the 2nd closest bowl.",
    "Pick the object and place in the 3rd closest bowl.",
    "Pick the object and place in the 4th closest bowl.",
    "Pick the object and place in the 5th closest bowl.",
    "Pick the object and place in the 6th closest bowl.",
    "Pick the object and place in the 7th closest bowl.",
    "Pick the object and place in the 1st furtherest bowl.",
    "Pick the object and place in the 2nd furtherest bowl.",
    "Pick the object and place in the 3rd furtherest bowl.",
    "Pick the object and place in the 4th furtherest bowl.",
    "Pick the object and place in the 5th furtherest bowl.",
    "Pick the object and place in the 6th furtherest bowl.",
    "Pick the object and place in the 7th furtherest bowl.",
    # allocentric pick: many same object + 1 bowl
    "Pick the object closest to the bowl and place in the bowl.",
    "Pick the object furtherest to the bowl and place in the bowl.",
    "Pick the 1st object closest to the bowl and place in the bowl.",
    "Pick the 2nd object closest to the bowl and place in the bowl.",
    "Pick the 3rd object closest to the bowl and place in the bowl.",
    "Pick the 4th object closest to the bowl and place in the bowl.",
    "Pick the 5th object closest to the bowl and place in the bowl.",
    "Pick the 6th object closest to the bowl and place in the bowl.",
    "Pick the 7th object closest to the bowl and place in the bowl.",
    "Pick the 1st object furtherest to the bowl and place in the bowl.",
    "Pick the 2nd object furtherest to the bowl and place in the bowl.",
    "Pick the 3rd object furtherest to the bowl and place in the bowl.",
    "Pick the 4th object furtherest to the bowl and place in the bowl.",
    "Pick the 5th object furtherest to the bowl and place in the bowl.",
    "Pick the 6th object furtherest to the bowl and place in the bowl.",
    "Pick the 7th object furtherest to the bowl and place in the bowl.",
    # allocentric place: 1 object + many bowls
    "Pick the object and place in the bowl closest to it.",
    "Pick the object and place in the bowl furtherest from it.",
    "Pick the object and place in the 1st bowl closest to it.",
    "Pick the object and place in the 2nd bowl closest to it.",
    "Pick the object and place in the 3rd bowl closest to it.",
    "Pick the object and place in the 4th bowl closest to it.",
    "Pick the object and place in the 5th bowl closest to it.",
    "Pick the object and place in the 6th bowl closest to it.",
    "Pick the object and place in the 7th bowl closest to it.",
    "Pick the object and place in the 1st bowl furtherest from it.",
    "Pick the object and place in the 2nd bowl furtherest from it.",
    "Pick the object and place in the 3rd bowl furtherest from it.",
    "Pick the object and place in the 4th bowl furtherest from it.",
    "Pick the object and place in the 5th bowl furtherest from it.",
    "Pick the object and place in the 6th bowl furtherest from it.",
    "Pick the object and place in the 7th bowl furtherest from it.",
    # pick by feature: many different object + 1 bowl
    "Pick the largest object and place in the bowl.",
    "Pick the smallest object and place in the bowl.",
    "Pick the 1st largest object and place in the bowl.",
    "Pick the 2nd largest object and place in the bowl.",
    "Pick the 3rd largest object and place in the bowl.",
    "Pick the 4th largest object and place in the bowl.",
    "Pick the 5th largest object and place in the bowl.",
    # "Pick the 6th largest object and place in the bowl.",
    # "Pick the 7th largest object and place in the bowl.",
    "Pick the 1st smallest object and place in the bowl.",
    "Pick the 2nd smallest object and place in the bowl.",
    "Pick the 3rd smallest object and place in the bowl.",
    "Pick the 4th smallest object and place in the bowl.",
    "Pick the 5th smallest object and place in the bowl.",
    # place by feature: 1 object + different bowl
    "Pick the object and place in the largest bowl.",
    "Pick the object and place in the smallest bowl.",
    "Pick the object and place in the 1st largest bowl.",
    "Pick the object and place in the 2nd largest bowl.",
    "Pick the object and place in the 3rd largest bowl.",
    "Pick the object and place in the 1st smallest bowl.",
    "Pick the object and place in the 2nd smallest bowl.",
    "Pick the object and place in the 3rd smallest bowl.",
    # middle pick: 3 different object with col restriction + one bowl
    "Pick the object in the middle and place in the bowl."
    "Pick the object and place in the bowl in the middle."
]



