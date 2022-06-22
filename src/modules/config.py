### IMPORTS ###

import os
import glob

### PARAMETERS ###

# data #
PERSON = 1
SECTIONS = [1, 4, 9]
COLORSPACE = 'grayscale'
STRIDE = 12
RESIZE_FACTOR = 10
RESIZE = (480/RESIZE_FACTOR, 640/RESIZE_FACTOR)

# folder paths #
BASE_PATH = f'../data/lpw/{PERSON}'
OUTPUT_PATH = f'../output'
ANNOT_PATHS = [f'../{BASE_PATH}/{SECTION}.txt' for SECTION in SECTIONS]
IMAGE_PATH = glob.glob(BASE_PATH + f'/*.jpg')
IMAGE_NAMES = [os.path.basename(IMAGE) for IMAGE in IMAGE_PATH]

# plotting #
MARKERCOLOR='green'
MARKERSIZE=15
MARKER='X'

# learning #
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 300

# data splitting #
SPLIT = 0.2