### IMPORTS ###

from modules.config import BASE_PATH, OUTPUT_PATH, ANNOT_PATHS, IMAGE_PATH, IMAGE_NAMES
from modules.config import PERSON, SECTIONS, COLORSPACE, STRIDE, RESIZE, RESIZE_FACTOR
from modules.config import SAMPLE_SIZE, MARKER, MARKERCOLOR, MARKERSIZE
from modules.config import BATCH_SIZE, SPLIT

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(style='darkgrid', palette='viridis')

from pyarrow import parquet, Table
import cv2

### HELPERS ###

def array_to_df(img_arrays):
    '''
    Converts image arrays to panda dataframes where multi-indexing is required.
    
    Args:
        img_arrays (array): multi-dimensional/nested image array

    Returns:
        pandas.DataFrame: multi-indexed DataFrame
    '''
    
    # define axis names and convent array to numpy for manipulations
    names = ['image', 'height', 'width']
    img_numpy = np.array(img_arrays)
    
    # iterate through shapes and assign names
    index = pd.MultiIndex.from_product(
        [range(dim) for dim in img_numpy.shape],
        names=names)

    # use multiindexing to configure the dataframe
    img_df = pd.DataFrame(
        { names[0] : img_numpy.flatten() },
        index=index)
    
    img_df = img_df['image']
    img_df = img_df.unstack(level='width').sort_index()
    
    return img_df

def df_to_array(df_images):
    
    # xarray conversion to get multi-index lengths
    dx_images = df_images.to_xarray()
    
    # reshape to image array from DataFrame
    img_array = df_images.values.reshape(
        dx_images.dims['image'], # number of images
        dx_images.dims['height'], # height, rows
        len(dx_images.data_vars), # width, cols
        ) # row level 1
    return img_array

### FUNCTIONS ###

def get_image_locs():
    '''
    Returns:
        pd.Series: image locations for a given person
    '''
    print('Getting image locations...')
    
    # image location urls
    image_locs = pd.Series(IMAGE_PATH, name='image_location')
    image_locs = image_locs.iloc[lambda x: x.index % STRIDE == 0]
        
    # add image urls to dataframe
    image_locs = image_locs.drop_duplicates().reset_index(drop=True)
    
    return image_locs

def get_images(save, result):
    '''
    Read the image, set the colorspace, and save if specified.

    Args:
        save (bool): saves to .parquet file if True
        result (string): 'df', 'dx', or 'array'

    Returns:
        array or pandas.DataFrame: returns image data converted into arrays
    '''

    # read images, set colorspace
    images = []
    print(f'Getting images in <{COLORSPACE}>...')
    
    if COLORSPACE == 'grayscale':
        for location in get_image_locs():
            image = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
            images.append(cv2.resize(image, (RESIZE)))
            
    elif COLORSPACE == 'alpha':
        for location in get_image_locs():
            image = cv2.imread(location, cv2.IMREAD_UNCHANGED)
            images.append(cv2.resize(image, (RESIZE)))
            
    else:
        for location in get_image_locs():
            image = cv2.imread(location)
            images.append(cv2.resize(image, (RESIZE)))
            
    print(f'Resizing images from {pd.DataFrame(image).shape} to {RESIZE}.\n')
    
    # convert array to dataframe
    df_images = array_to_df(images)
    dx_images = df_images.to_xarray()
    
    # save to file if set to True
    if save==True:
        table = Table.from_pandas(df_images.loc[(range(BATCH_SIZE), ), :])
        parquet.write_table(table, f'{OUTPUT_PATH}/P{PERSON}_imgs.parquet')
        print(f'Saved image dataframe to {OUTPUT_PATH}/P{PERSON}_imgs.parquet.\n')
    
    # return dataframe, xarray, or image array
    # use df for plotting, hence SAMPLE_SIZE
    if result=='df':
        df_images = df_images[df_images.index.get_level_values(0) < SAMPLE_SIZE]
        return df_images
    # use dx for training, BATCH_SIZE
    elif result=='dx':
        dx_images = dx_images.loc[dict(image=[idx for idx in range(BATCH_SIZE)])]
        return dx_images
    # use array for quick visuals
    else:
        images = images[:BATCH_SIZE]
        return images

def get_image_names():
    '''
    Returns:
        pd.Series: image names for a given person
    '''
    print('Getting image filenames...')
    
    # get image location paths and add to DataFrame
    image_names = pd.Series(IMAGE_NAMES, name='image_name')
    image_names= image_names.drop_duplicates().reset_index(drop=True)
    
    return image_names

def get_keypoints(with_image_locs, scale, save):
    '''
    Create dataframe of annotations relating to the given a person and the defined sections.

    Args:
        with_image_locs (bool): add image location column if True
        scale (bool): scale values using resize factor
        save (bool): saves to .parquet file if True

    Returns:
        dataframe: returns keypoints in a DataFrame
    '''
    
    # storage
    df_keypoints = pd.DataFrame()
    
    # interate for each annotation section
    for ANNOT_PATH in ANNOT_PATHS:
        # create new table to store keypoints
        df_new = pd.read_table(ANNOT_PATH, sep=' ', header=None, names=['x', 'y'])
        print(f'Reading data from {ANNOT_PATH}...')
        
        # add to existing keypoints dataframe
        df_keypoints = pd.concat( [df_keypoints, df_new] ).reset_index(drop=True)
    
    if with_image_locs==True:
        # add image locations to dataframe
        df_keypoints = pd.concat( [df_keypoints, get_image_names() ], axis=1)
        print(f'... and add column of image filenames to keypoint DataFrame.')
    
    print(f'Reducing keypoint data from ({len(df_keypoints.index)})...')
    
    df_keypoints = df_keypoints.iloc[lambda x: x.index % STRIDE == 0]
    df_keypoints_sampled = df_keypoints[:BATCH_SIZE]
    df_keypoints_sampled = df_keypoints_sampled.drop_duplicates().reset_index(drop=True)
    
    if BATCH_SIZE > len(df_keypoints.index):
        print(f'... to ({len(df_keypoints.index)})\n')
    else:
        print(f'... to ({BATCH_SIZE})\n')
    
    if scale==True:
        df_keypoints_sampled[['x','y']] = df_keypoints_sampled[['x','y']].apply(
            lambda x: x/RESIZE_FACTOR)
    
    # save annotation dataframe to file (parquet)
    if save==True:
        table = Table.from_pandas(df_keypoints_sampled)
        parquet.write_table(table, f'{OUTPUT_PATH}/P{PERSON}_kpts.parquet')
        print(f'Saved keypoint dataframe to {OUTPUT_PATH}/P{PERSON}_kpts.parquet.\n')
    
    return df_keypoints_sampled

def show_image_keypoint(imgs, keypoints, window, save):
    '''
    Visualize one image with its keypoint.

    Args:
        imgs (pd.DataFrame): image dataset
        keypoints (pd.DataFrame): keypoint dataset
        window (hashable): define start and stop window for slice sampling the datasets
        save (bool): save to file if True
    '''
    start = window[0]
    end = window[1]
    
    imgs = imgs[start:end]
    keypoints = keypoints[start:end]
    
    for i, (img, kps) in enumerate(zip(imgs, keypoints.iterrows())):
        
        _, ax = plt.subplots(figsize=(15,12))
        
        ax.axis('off')
        ax.imshow(img)
        ax.plot(kps[1][0], kps[1][1], color=MARKERCOLOR, marker=MARKER, markersize=MARKERSIZE)
    
        if save==True:
            cv2.imwrite(f'{OUTPUT_PATH}/figures/GIF_train/{kps[1][2]}', img)
    
    plt.tight_layout()
    plt.show()

def plot_image_keypoints(imgs, keypoints, pivot):
    '''
    Plot images with keypoints. Differs from show_image_keypoint because this outputs
    a grid visual of multiple images.

    Args:
        imgs (pd.DataFrame): images in DataFrame
        keypoints (pd.DataFrame): keypoints in DataFrame
        pivot (string): 'col' or 'row'; think of it as an anchor
        
    '''
    if pivot=='col':
        # define plot
        row = 0
        ncols = 2
        _, ax = plt.subplots(figsize=(15,30), nrows=int(len(imgs)/2), ncols=ncols)
        
        for i, (img, kps) in enumerate(zip(imgs, keypoints.iterrows())):
            
            # subplot rows and columns
            col = i % 2
            
            # column 0
            if col == 0:
                # print(col,row)
                ax[row,col].axis('off')
                ax[row,col].imshow(img)
                ax[row,col].plot(kps[1][0], kps[1][1], color=MARKERCOLOR, marker=MARKER, markersize=MARKERSIZE)
            # column 1
            else:
                # print(col,row)
                ax[row,col].axis('off')
                ax[row,col].imshow(img)
                ax[row,col].plot(kps[1][0], kps[1][1], color=MARKERCOLOR, marker=MARKER, markersize=MARKERSIZE)
                # change row
                row += 1
                
        plt.tight_layout()
        plt.show()
    
    if pivot=='row':
        # define plot
        col = 0
        nrows = 2
        _, ax = plt.subplots(figsize=(30,12), ncols=int(len(imgs)/2), nrows=nrows)

        for i, (img, kps) in enumerate(zip(imgs, keypoints.iterrows())):
            
            # subplot rows and columns
            row = i % 2
            
            # row 0
            if row == 0:
                # print(col,row)
                ax[row,col].axis('off')
                ax[row,col].imshow(img)
                ax[row,col].plot(kps[1][0], kps[1][1], color=MARKERCOLOR, marker=MARKER, markersize=MARKERSIZE)
            # row 1
            else:
                # print(col,row)
                ax[row,col].axis('off')
                ax[row,col].imshow(img)
                ax[row,col].plot(kps[1][0], kps[1][1], color=MARKERCOLOR, marker=MARKER, markersize=MARKERSIZE)
                # change row
                col += 1
                
        plt.tight_layout()
        plt.show()

def save_loss_plots(history):
    """
    Plot and save resulting losses from model training.
    
    Args:
        history (method): results from model training

    Returns:
        None
    """
    train_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    
    # loss plot
    plt.figure(figsize=(15, 10))
    plt.plot(
        train_loss,
        color='blue',
        linestyle='=', 
        label='Train Loss')
    plt.plot(
        validation_loss,
        color='red',
        linestyle='-', 
        label='Validation Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.savefig(f'{OUTPUT_PATH}/figures/loss_plot.png')
    print(f'Saved figure to {OUTPUT_PATH}/figures/loss_plot.png')
    
    plt.show()



# Adding leading zeroes to the image name:
# {0 : 0 > 4}
#  │   │ │ │
#  │   │ │ └─ Width of 4
#  │   │ └─ Align Right
#  │   └─ Fill with '0'
#  └─ Element index / element
# .drop_duplicates().reset_index(drop=True).sort_values()