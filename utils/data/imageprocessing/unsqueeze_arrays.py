##########################################################################################################################################
#Script that will unsqueeze a large number of arrays for us. Especially important if we have images of dimension (H, W) and we need this #
#   to be (C, H, W)                                                                                                                      #
##########################################################################################################################################

#Module imports 
import numpy as np
import os

def unsqueeze_arrays(data_path:str):
    """Unsqueezes all arrays at a specified location (path)
    
    Parameters
    ----------
        data_path : str
            The path to the location where the .npy arrays (which we want to unsqueeze) are all stored.
    """

    #List of the individual array file names and paths in the dataset
    array_names = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    array_paths = [os.path.join(data_path, n) for n in array_names]

    #Load the arrays one-by-one
    for array_path in array_paths:
        arr = np.load(array_path)
        arr = arr[None, :]
        np.save(file=array_path, arr=arr) #save the updated array at original location


if __name__ == "__main__":
    data_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/test set'
    unsqueeze_arrays(data_path=data_path)

