import os

import pandas as pd

NUM_CLASSES = {
    'Touch_and_Go': {
        'material': 20,
        'rough': 2,
        'hard': 2
    },
    'feeling':{
        'grasping': 2
    },
    'ObjectFolderReal': {
        'material': 7
    },
}


def check_value(c):
    if c == 'True':
        return 'the object being touched is lifted in the air.'
    elif c == 'False':
        return 'the object being touched is falling on the ground.'
    else:
        raise ValueError('Invalid input: must be "True" or "False"')


OPENAI_IMAGENET_TEMPLATES = {
    'material': (
        lambda c: f'the object being touched is {c}, it is an image of {c}.',
        # lambda c: f"a tactile image of {c}.",
    ),
    'rough': (
        lambda c: f"the object being touched feels {c}, the surface show sort of {c}ness.", 
        # lambda c: f"it feels {c}, this type of material is {c}.",
    ),
    'hard': (
        lambda c: f"the object being touched feels {c} , the surface show sort of {c}ness when touched.", 
        # lambda c: f"this type of material is {c}.",
    ),
    'grasping': (
        lambda c: check_value(c), 
        # lambda c: f"this type of material is {c}.",
    ),

}



CLASSNAMES = {
    'Touch_and_Go': {
        'material':  (
            "brick", "concrete", "glass", "grass", "gravel", "leather", "metal", "others", "paper", "plants",
            "plastic", "rock", "sand", "soil", "synthetic fabric", "tile", "tree", "wood", "rubber", "natural fabric"
        ),
        'rough': (
            "rough", "smooth"
        ),
        'hard': (
            "hard", "soft"
        )
    },
    'feeling': {
        'grasping':(
            "True", "False"
        )
    } 
}

