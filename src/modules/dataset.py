from modules.config import BASE_PATH, OUTPUT_PATH, ANNOT_PATHS, IMAGE_PATH, IMAGE_NAMES
from modules.config import PERSON, SECTIONS, COLORSPACE, STRIDE, RESIZE, RESIZE_FACTOR
from modules.config import BATCH_SIZE, SPLIT
from modules import utils
from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(style='darkgrid')

from pyarrow import parquet, Table
import cv2

def get_data(save):
    
    # get images and target outputs, then convert image to dataframe and save (~45min)
    img_array = utils.get_images(image_set=PERSON, scale=True, save=False, result='array')
    df_keypoints = utils.get_keypoints(with_image_locs=True, scale=True, save=False)

    # X image arrays, y keypoint coordinates
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        np.array(img_array),
        df_keypoints[['x', 'y']],
        test_size=SPLIT)
    
    print(f'SUMMARY\n \
        ++++++++++\n \
        Xtrain:\t{len(Xtrain)}\n \
        Xtest:\t{len(Xtest)}\n \
        ytrain:\t{len(ytrain)}\n \
        ytest:\t{len(ytest)}\n')
    
    # save training/validation sets to file (parquet)
    if save==True:
        for name, dataset in {'Xtrain':Xtrain, 'Xtest':Xtest, 'ytrain':ytrain, 'ytest':ytest}.items():
            # convert X data from array to DataFrame
            if name == 'Xtrain' or name == 'Xtest':
                dataset = utils.array_to_df(dataset)
            
            table = Table.from_pandas(dataset)
            parquet.write_table(table, f'{OUTPUT_PATH}/train_val/P{PERSON}_{name}.parquet')
            print(f'Saved {name} dataframe to {OUTPUT_PATH}/train_val/P{PERSON}_{name}.parquet.')
    
    return Xtrain, Xtest, ytrain, ytest