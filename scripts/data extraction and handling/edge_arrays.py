##########################################################
# Produces feature maps of the edges in the TDYNO Images #
# AND saves as .npy arrays                               #
##########################################################
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def normalise_array(array:np.ndarray, shot_name:str):
    """Normalises some input array by scaling to the maximum value.
    
    Parameters
    ----------
        array : np.ndarray
            The image array to be normalised.
        shot_name : str
            The name of the shot whose image we are processing.

    Returns
    -------
        normalised_array : np.ndarray
            The image array after normalisation.
    
    """

    #Catch instances of NaNs inside of the image arrays
    if np.isnan(array).any():
        raise ValueError(f"Warning: NaN numbers encountered in {shot_name} image array.")
    
    if (array < 0).any():
        raise ValueError(f"Warning: Negative values encountered in {shot_name} image")
    
    #Find max value and divide whole array by this value
    max_value = np.max(array)
    normalised_array = np.multiply(array, 1/max_value)
    
    return normalised_array

############################
# Edge Detection Functions #
############################

def horizontal_edges(image_directory:str, filter, filter_key):
    """Looks for horizontal edges in all shot images.

    Parameters
    ----------
        image_directory : str
            Path to the background-corrected shot image arrays
        filter : torch.tensor
            Filter kernel 
        filter_key : str
            Tells us which filter we are using
    
    """

    #List all of the shots in the image directory, which should be .npy
    shot_list = [f for f in os.listdir(image_directory) if f.endswith('.npy')]

    #Dictionary containing "edges" image for each shot
    edges_dict = {}
    
    for shot_file in shot_list:
        #Strip the '.npy' from the shot filename
        shot_name = shot_file[:-4]

        #Load the individual shot arrays, normalise.
        shot_path = os.path.join(image_directory, shot_file)
        shot_array = np.load(shot_path)
        normalised_array = normalise_array(shot_array, shot_name=shot_name)

        #Pass to torch tensor for convolution operation
        shot_tensor = torch.tensor(normalised_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        #Filter for horizontal edges using 2D convolution
        edges = torch.abs(F.conv2d(shot_tensor, weight=filter, padding=1, stride=1))
        edges_dict[shot_name] = edges

        #Set the save location for each array, and then save it.
        save_path = f'./data/processed/horizontal_edge_detector_sets/{shot_name}_{filter_key}'
        np.save(file=save_path, arr=edges)

        

########
# MAIN #
########

def main():
    #Path to the numpy arrays of the background-corrected shot images
    image_directory = './data/raw/tdyno2022_shot_data/corrected_image_arrays'

    #Filters dictionary to keep track of the various types of filters that we want to use. 
    filters = {
        "identity": torch.tensor([[0,0,0], [0,1,0], [0,0,0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "sobel_1": torch.tensor([[-1,-1,-1], [0,0,0], [1,1,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "sobel_2": torch.tensor([[-1,-2,-1], [0,0,0], [1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        "laplace": torch.tensor([[0,1,0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    }

    for filter_key in filters:
        horizontal_edges(image_directory=image_directory, filter=filters[filter_key], filter_key=filter_key)



if __name__ == '__main__':
    main()