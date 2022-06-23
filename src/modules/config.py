### IMPORTS ###

import os
import glob

### PARAMETERS ###

# data #
PERSON = 1
SECTIONS = [1, 4, 9]
COLORSPACE = 'grayscale'
STRIDE = 12
RESIZE_FACTOR = 5
RESIZE = (int(640/RESIZE_FACTOR), int(480/RESIZE_FACTOR))

# folder paths #
BASE_PATH = f'../data/lpw/{PERSON}'
OUTPUT_PATH = f'../output'
ANNOT_PATHS = [f'{BASE_PATH}/{SECTION}.txt' for SECTION in SECTIONS]
IMAGE_PATH = glob.glob(BASE_PATH + f'/*.jpg')
IMAGE_NAMES = [os.path.basename(IMAGE) for IMAGE in IMAGE_PATH]

# plotting #
SAMPLE_SIZE = 10
MARKERCOLOR = 'green'
MARKERSIZE = 20
MARKER = 'X'

# learning #
BATCH_SIZE = 500
LR = 0.001
EPOCHS = 500

# data splitting #
SPLIT = 0.2