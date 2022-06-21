# IMPORTS #

from modules.config import BASE_PATH, FILE_PATH, OUTPUT_PATH

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# FUNCTIONS #

def import_annot(person, sections, ext='txt', save=False):
    '''
    import_annot
    Create dataframe of annotations relating to the given a person
    and the defined sections.

    Args:
        person (int): person ID
        sections (list): photo set IDs for each person
        ext (str, optional): file extension for output; defaults to 'txt'
        save (bool, optional): save to csv

    Returns:
        dataframe: dataframe modified with the 
    '''
    
    df = pd.DataFrame()
    base = BASE_PATH(person)
    out = OUTPUT_PATH
    
    for section in sections:
        annotFile = FILE_PATH(
            base=base,
            section=section,
            kind='annot',
            ext=ext)
        
        # create new table
        new_df = pd.read_table(
            annotFile,
            sep=' ',
            header=None,
            names=['x', 'y'])
        
        # add to existing table
        df = pd.concat([df, new_df]).reset_index(drop=True)
        
    imageNames = pd.Series(
        FILE_PATH(
            base=base,
            kind='image',
            ext='jpg',
            path=False),
        name='image_name')
    
    imageNames = imageNames.drop_duplicates().reset_index(drop=True)
    df = pd.concat([df, imageNames], axis=1)
    
    if save==True:
        df.to_csv(os.path.sep.join([out, f'person{person}_kpts.csv']))
    
    return df

def array_to_pandas(img_arrays):
    
    # define axis names and convent array to numpy for manipulations
    names = ['image', 'height', 'width']
    img_numpy = np.array(img_arrays)
    
    # iterate through shapes and assign names
    index = pd.MultiIndex.from_product(
        [range(s) for s in img_numpy.shape], names=names)

    # use multiindexing to configure the dataframe
    df = pd.DataFrame(
        { names[0] : img_numpy.flatten() },
        index=index)
    df = df['image']
    df = df.unstack(level='width').sort_index()
    
    return df

def convert_image(imgs, color, save=False, pathname=None, to_df=False):
    '''
    image_to_array
    Read the image, set the colorspace, and save if specified.

    Args:
        imgs (path/filename): specify imgs to read; list of urls or paths
        color (string): define the colorspace; 'grayscale', 'color', or 'alpha'
        save (bool, optional): save to file; defaults to False
        pathname (string, optional): define save location; defaults to None
        to_df (bool, optional): specify if it should return a dataframe

    Returns:
        array: returns image data converted into arrays
    '''
    
    # read images, set colorscale
    images = []
    if color == 'grayscale':
        for img in imgs:
            images.append(cv2.imread(img, cv2.IMREAD_GRAYSCALE))
    elif color == 'alpha':
        for img in imgs:
            images.append(cv2.imread(img, cv2.IMREAD_UNCHANGED))
    else:
        for img in imgs:
            images.append(cv2.imread(img))
    
    df_images = array_to_pandas(images)
    
    # save to file if set to True
    if save==True:
        df_images.to_csv(pathname, ',')
    
    # return dataframe instead of 
    if to_df==True:
        return df_images
    
    return images

def plot_image_keypoints(imgs, keypoints, markercolor='green', markersize=15, marker='X'):
            
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
            ax[row, col].plot(kps[1][0], kps[1][1], color=markercolor, marker=marker, markersize=markersize)
        # column 1
        else:
            # print(col,row)
            ax[row,col].axis('off')
            ax[row,col].imshow(img)
            ax[row, col].plot(kps[1][0], kps[1][1], color=markercolor, marker=marker, markersize=markersize)
            # change row
            row += 1
            
    plt.tight_layout()
    plt.show()

def save_loss_plots(history):
    """
    save_loss_plots
    Plot and save loss plots
    
    Args:
        history (_type_): _description_

    Returns:
        None
    """
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    
    # Loss plots.
    plt.figure(figsize=(15, 10))
    plt.plot(
        train_loss, color='blue', linestyle='=', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.savefig(
        os.path.join('..', 'output', 'figures', 'loss.png'))
    
    plt.show()

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