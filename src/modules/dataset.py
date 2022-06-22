from modules import config, utils, model
import cv2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(style='darkgrid')

from sklearn.model_selection import train_test_split

def get_data():
    
    # get images, then convert image to dataframe and save (~45min)
    df_images = utils.get_images(save=True, return_df=False)

    # target dataframe and output
    df_keypoints = utils.get_keypoints(with_image_locs=True, save=True)

    # X image arrays, y keypoint coordinates
    Xtrain, Xtest, ytrain, ytest = train_test_split(df_images, df_keypoints[['x', 'y']], test_size=config.SPLIT)
    
    return Xtrain, Xtest, ytrain, ytest