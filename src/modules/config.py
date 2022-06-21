# IMPORTS #

import os
import glob


# VARIABLES #

def BASE_PATH(person):
    '''
    BASE_PATH
    Define base path to be used in other pathfinding functions.
    Person of interest is provided explicitly.

    Args:
        person (int): define the person or subject

    Returns:
        string: base path
    '''
    basePath = f'../data/lpw/{person}'
    
    return basePath

def FILE_PATH(base, section=None, kind=None, ext=None, path=True):
    '''
    FILE_PATH
    Gets annotations for the given filepath.
    OR
    Gets images for the given filepath.

    Args:
        base (string): path of project base
        ext (string): file extension

    Returns:
        string: returns filepath (to image or annot) string
    '''
    
    # for annotations
    if kind=='annot':
        annotFile = os.path.sep.join([base, f'{section}.{ext}'])
        return annotFile
    
    # for images
    elif kind=='image':
        imagePaths = glob.glob(base + f'/*.{ext}')
        if path==True:
            return imagePaths
        else:
            imageNames = [os.path.basename(x) for x in imagePaths]
            return imageNames
    
    else:
        return " 'kind' not recognized. Use either 'annot' or 'image'."


# PARAMETERS #

# folder structure #
OUTPUT_PATH = os.path.sep.join(['..', 'output'])

# learning #
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 300

# data splitting #
SPLIT = 0.2