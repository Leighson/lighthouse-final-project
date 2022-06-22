from modules import config, utils, model
import cv2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(style='darkgrid')

from sklearn.model_selection import train_test_split

def get_data():
    
    # get images and target outputs, then convert image to dataframe and save (~45min)
    img_array = utils.get_images(save=False, result='array')
    df_keypoints = utils.get_keypoints(with_image_locs=True, scale=True, save=False)
    # print(f'dx_images: {img_array}, df_keypoints: {df_keypoints}')

    # X image arrays, y keypoint coordinates
    Xtrain, Xtest, ytrain, ytest = train_test_split(img_array, df_keypoints[['x', 'y']], test_size=config.SPLIT)
    
    print(f'SUMMARY\nXtrain: {len(Xtrain)}\nXtest: {len(Xtest)}\nytrain: {len(ytrain)}\nytest: {len(ytest)}')
    
    return Xtrain, Xtest, ytrain, ytest