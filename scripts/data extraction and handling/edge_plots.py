##########################################################
# Produces feature maps of the edges in the TDYNO Images #
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

def plot_edges_images(edges_dict:dict, orientation:str, filter_key:str):
    """Produces plots of the feature maps produced via convolution with the filter kernel.

    Parameters
    ----------
        edges_dict : dict
            Dictionary containing the feature maps from the convolution for each shot image.
        orientation : str
            String telling us whether the edges are vertical or horizontal
        filter_key : str
            Tells us which type of filter we are using in order to give informative plot labels.
    
    """
    
    keys = list(edges_dict.keys())
    edge_tensors = list(edges_dict.values())
    
    #Set the number of columns for the subplots to 2, and calculate how many rows we need.
    ncols = 4
    nrows = int(np.ceil(len(edges_dict) / ncols))
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,15))
    
    #Iterate over shot images and shot labes, plot
    for i in range(nrows):
        for j in range(ncols):
            shot_index = i*ncols + j
            if shot_index < len(keys):

                #Call the image torch tensor and convert to numpy array
                image_array = edge_tensors[shot_index][0][0].numpy() #account for batch/filter no.
                shot_title = keys[shot_index]
                
                #PLOT THE IMAGES
                axs[i, j].imshow(image_array, cmap='inferno')
                axs[i, j].set_title(shot_title) 
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

    fig.suptitle(f'{orientation} Edges for Shot Images, {filter_key} Filter')
    fig.tight_layout()
    plt.savefig(f'./plots/{orientation}_edges/{filter_key}/horizontal_edges_all_plots.png')
    plt.show()

    for i, image_tensor in enumerate(edge_tensors):
        shot_name = keys[i]
        plt.imshow(image_tensor[0, 0].numpy(), cmap='inferno')
        plt.title(f'{shot_name} {orientation} Edges, {filter_key} Filter')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'./plots/{orientation}_edges/{filter_key}/{shot_name}.png')
        plt.show()
    



def horizontal_edges(image_directory:str, filter, filter_key, plot:bool=False):
    """Looks for horizontal edges in all shot images.

    Parameters
    ----------
        image_directory : str
            Path to the background-corrected shot image arrays
        filter : torch.tensor
            Filter kernel 
        filter_key : str
            Tells us which filter we are using
        plot : bool
            Determines whether the user wishes to produce plots of the vertical edges for 
    
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

    if plot:
        plot_edges_images(edges_dict=edges_dict, orientation='horizontal', filter_key=filter_key)

        

########
# MAIN #
########

def main(plot=True):
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
        horizontal_edges(image_directory=image_directory, filter=filters[filter_key], filter_key=filter_key, plot=plot)



if __name__ == '__main__':
    main(plot=True)