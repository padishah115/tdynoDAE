###########
# After extracting the raw images fro .hdf files and saving them as arrays, we need to make sure
# that we don't have any phantom NaNs, etc. The raw image arrays are stored in ./data/raw/tdyno2022_shot_data/raw_image_arrays

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

###################################################################################
# Function for loading the image arrays, correcting for background, then removing #
###################################################################################

def load_image(arr_path:str, shot_no:int):
    """Function handling our loading and plotting of streak images from arrays.
    
    Parameters
    ----------
        arr_path : str
            Path to the numpy array where the streak data is stored.
        shot_no : int
            The shot number for this particular data
    """

    print(f"\nAccessing raw image array at {arr_path} \n...")
    arr = np.load(arr_path)
    print(f"Successfully loaded data for shot {shot_no}. Array shape {arr.shape}")
    

    #Separate the background image from the shot data, and subtract the two images to get the shot data.
    background_image = arr[1]
    shot_image = arr[0]
    corrected_image = shot_image-background_image
    
    #Set lower threshold to 0
    corrected_image[corrected_image<0] = 0
    #print(f'Max pixel: {np.nanmax(corrected_image)}, Min pixel: {np.nanmin(corrected_image)}')

    return corrected_image


###########################
# Functions for PLOTTING. #
###########################

def plot_together(raw_images_path, raw_images_list):
    """Plots the background-corrected shot data images together on a single master plot, and saves the master plot.

    Parameters
    ----------
        raw_images_path : str
            The path to the raw image arrays containing the background data AND shot data.
        raw_images_list : list
            List of shot files (expect '.npy' extension)
    
    """

    #Configure the subplots element.
    ncols = 3
    nrows = int(np.ceil(len(raw_images_list) / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))

    for i, raw_image_filename in enumerate(raw_images_list[:]):
        shot_no = raw_image_filename[:-4] #remove '.npy' from shot filename to get shot number
        im_path = os.path.join(raw_images_path, raw_image_filename) #Get path to the raw image array containing shot AND background
        
        #Subtract background from shot image
        corrected_image = load_image(arr_path=im_path, shot_no=shot_no)
        
        #Plot each corrected image in the FIGURE element, each in its own subplot
        rownum = int(np.floor(i / ncols))
        colnum = int(i % ncols)
        axs[rownum, colnum].imshow(corrected_image, cmap='inferno')
        axs[rownum, colnum].set_title(shot_no)
        axs[rownum, colnum].set_xticks([])
        axs[rownum, colnum].set_yticks([])

    #Plot master graph and save.
    fig.suptitle('Background-Corrected Images: \nAll Shots')
    fig.savefig('./plots/background_corrected_images_all_shots.png')
    plt.show()



def plot_individual(raw_images_path:str, raw_images_list:list):
    """Plots the background-corrected shot data images individually, and saves each shot image as a .png.

    Parameters
    ----------
        raw_images_path : str
            The path to the raw image arrays containing the background data AND shot data.
        raw_images_list : list
            List of shot files (expect '.npy' extension)

    Notes
    -----
        Does not display the individual plots before saving.
    
    """


    for i, raw_image_filename in enumerate(raw_images_list[:]):
        shot_no = raw_image_filename[:-4] #remove '.npy' from shot filename to get shot number
        im_path = os.path.join(raw_images_path, raw_image_filename) #Get path to the raw image array containing shot AND background
        
        #Subtract background from shot image
        corrected_image = load_image(arr_path=im_path, shot_no=shot_no)

        #Save individual (correct) shot images as numpy arrays
        save_path = os.path.abspath(f'./data/raw/tdyno2022_shot_data/corrected_image_arrays/{shot_no}')
        np.save(file=save_path, arr=corrected_image)

        #Show background-corrected BITMAP images, and save figure
        plt.imshow(corrected_image, cmap='inferno')
        plt.title(shot_no)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'./plots/background_removed/{shot_no}.png')
        #plt.show() #I don't want to show the plots one-by-one at the moment.

########
# MAIN #
########

def main():
    #Load up the path where the raw image arrays are stored and take a look at what we have
    relative_raw_images_path = './data/raw/tdyno2022_shot_data/raw_image_arrays'
    raw_images_path = os.path.abspath(relative_raw_images_path)
    raw_images_list = os.listdir(raw_images_path) #List of raw images

    #Produce the individual and master plots.
    plot_together(raw_images_path=raw_images_path, raw_images_list=raw_images_list)
    plot_individual(raw_images_path=raw_images_path, raw_images_list=raw_images_list)

if __name__ == "__main__":
    main()

