### IMPORTS ###

from modules.config import BASE_PATH, OUTPUT_PATH, ANNOT_PATHS, IMAGE_PATH, IMAGE_NAMES
from modules.config import PERSON, SECTIONS, COLORSPACE, STRIDE, RESIZE
from modules.config import MARKER, MARKERCOLOR, MARKERSIZE

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(style='darkgrid', palette='viridis')

from pyarrow import parquet, Table
import cv2
import pyarrow as pa

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

### FUNCTIONS ###

def get_image_locs():
    '''
    Returns:
        pd.Series: image locations for a given person
    '''
    
    # image location urls
    image_locs = pd.Series(IMAGE_PATH, name='image_location')
    image_locs = image_locs.iloc[lambda x: x.index % STRIDE == 0]
        
    # add image urls to dataframe
    image_locs = image_locs.drop_duplicates().reset_index(drop=True)
    
    return image_locs

def get_images(save, return_df):
    '''
    Read the image, set the colorspace, and save if specified.

    Args:
        save (bool): saves to .parquet file if True
        return_df (bool): True returns a dataframe; False returns an array

    Returns:
        array or pandas.DataFrame: returns image data converted into arrays
    '''

    # read images, set colorspace
    images = []
    print('Getting images in <{COLORSPACE}>...')
    
    if COLORSPACE == 'grayscale':
        for location in get_image_locs():
            image = cv2.imread(location, cv2.IMREAD_GRAYSCALE)
            images.append(cv2.resize(image, RESIZE))
            
    elif COLORSPACE == 'alpha':
        for location in get_image_locs():
            image = cv2.imread(location, cv2.IMREAD_UNCHANGED)
            images.append(cv2.resize(image, RESIZE))
            
    else:
        for location in get_image_locs():
            image = cv2.imread(location)
            images.append(cv2.resize(image, RESIZE))
    
    # convert array to dataframe
    df_images = array_to_df(images)
    
    # save to file if set to True
    if save==True:
        # df_images.to_csv(f'{OUTPUT_PATH}/P{PERSON}_imgs.csv', ',')
        table = Table.from_pandas(df_images)
        parquet.write_table(table, f'{OUTPUT_PATH}/P{PERSON}_imgs.parquet')
    
    # return dataframe instead of image array
    if return_df==True:
        return df_images
    
    return images

def get_image_names():
    '''
    Returns:
        pd.Series: image names for a given person
    '''
    print('Getting image pathnames...')
    
    # get image location paths and add to DataFrame
    image_names = pd.Series(IMAGE_NAMES, name='image_name')
    image_names= image_names.drop_duplicates().reset_index(drop=True)
    
    return image_names

def get_keypoints(with_image_locs, save):
    '''
    Create dataframe of annotations relating to the given a person and the defined sections.

    Args:
        with_image_locs (bool): add image location column if True
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
    
    df_keypoints = df_keypoints.iloc[lambda x: x.index % STRIDE == 0]
    
    if with_image_locs==True:
        # add image locations to dataframe
        df_keypoints = pd.concat( [df_keypoints, get_image_names() ], axis=1)
    
    # save annotation dataframe to file (parquet)
    if save==True:
        # df_keypoints.to_csv(f'{OUTPUT_PATH}/p{PERSON}_kpts.csv', ',')
        table = Table.from_pandas(df_keypoints)
        parquet.write_table(table, f'{OUTPUT_PATH}/P{PERSON}_kpts.parquet')
        print(f'Saved dataframe to {OUTPUT_PATH}/P{PERSON}_kpts.parquet')
    
    return df_keypoints

def plot_image_keypoints(imgs, keypoints):
            
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
            ax[row, col].plot(kps[1][0], kps[1][1], color=MARKERCOLOR, marker=MARKER, markersize=MARKERSIZE)
        # column 1
        else:
            # print(col,row)
            ax[row,col].axis('off')
            ax[row,col].imshow(img)
            ax[row, col].plot(kps[1][0], kps[1][1], color=MARKERCOLOR, marker=MARKER, markersize=MARKERSIZE)
            # change row
            row += 1
            
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



# os.path.sep.join([out, f'person{person}_img.csv']

# def image_to_array(images, save=True, pathname=None):
#     '''
#     image_to_array
#     Converts image data to arrays

#     Args:
#         images (pandas.Series): data series of indexed image pathnames

#     Returns:
#         _type_: _description_
#     '''
#     X = np.array([np.array(Image.open(img), 'float') for img in images])
    
#     # save to file
#     if save==True:
#         X.tofile(pathname, sep=',')
    
#     return X

# Adding leading zeroes to the image name:
# {0 : 0 > 4}
#  │   │ │ │
#  │   │ │ └─ Width of 4
#  │   │ └─ Align Right
#  │   └─ Fill with '0'
#  └─ Element index / element
# .drop_duplicates().reset_index(drop=True).sort_values()