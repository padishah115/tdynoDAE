#########################################################################
# SCRIPT FOR TRIMMING IMAGES IN TRAINING AND VALIDATION SETS TO SQUARES #
#########################################################################

#Standard imports
import os
import numpy as np


def trim_dataset(untrimmed_dataset_path:str, trimmed_dataset_path:str, im_width:int=2064):
    """Trims all of the files in a certain location to the specified image width, and saves in a new directory.

    Parameters 
    ----------
        dataset_path : str
            The source path containing the untrimmed dataset images.
        trimmed_dataset_path : str
            The destination path where we want to save the dataset images after they have been trimmed
        im_width : int
            The dimensions of the image, in pixels, that we want the image to have after trimming.
    
    """

    ###################
    #TRIM DATA SET    #
    ###################

    #Scan the specified directory looking for .npy files that we will trim
    untrimmed_set = [f for f in os.listdir(untrimmed_dataset_path) if f.endswith('.npy')]
    for filename in untrimmed_set:
        array_load_path = os.path.join(untrimmed_dataset_path, filename)
        array = np.load(file=array_load_path)

        #KEEP ONLY THE DESIRED ENTRIES IN ORDER TO MAKE SQUARE
        array = array[:im_width, :im_width]
        print(filename, array.shape)

        #SAVE THE TRIMMED IMAGES IN THE TRIMMED DIRECTORIES
        array_save_path = os.path.join(trimmed_dataset_path, filename)
        np.save(file=array_save_path, arr=array)

    
if __name__ == '__main__':
    untrimmed_path = './data/raw/tdyno2022_shot_data/corrected_image_arrays'
    trimmed_path = "./data/processed/horizontal_edge_detector_sets/data_augmentation_method/test set"
    trim_dataset(
        untrimmed_dataset_path=untrimmed_path,
        trimmed_dataset_path=trimmed_path
    )


