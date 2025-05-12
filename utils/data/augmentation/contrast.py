################################################################
# UTILITY FOR ARTIFICALLY INCREASING THE CONTRAST OF THE IMAGE #
################################################################

#Module imports
import numpy as np
import os

def add_contrast_to_dataset(source_path:str, dest_path:str, sat_val:float, alpha:float):
    """Adds contrast to all of the arrays in some dataset at a location supplied by the source_path variable.
    
    Parameters
    ----------
        source_path : str
            The path where the original (unnoised) .npy arrays are stored.
        dest_path : str
            The path where we want to save the arrays after contrast has been added.
        sat_val : float
            The value of pixel intensity at which we are considering the pixel to be saturated. Typically, this is the max measured intensity.
        alpha : float
            Linear scaling factor that we will use to add image contrast.
        noising_index : int
            The iteration number tracking how many times we have added contrast at this alpha value to the test images.

    """
    
    #Get the individual arrays' paths in the dataset directory
    array_load_paths = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.npy')]
    
    #Modify the names of the array locations to reflect the noising process
    new_array_names = [f[:-4]+f'-contrast_alpha-{alpha}' for f in os.listdir(source_path) if f.endswith('.npy')]
    
    #Also generate the 'save' pathnames in advance
    destination_paths = [os.path.join(dest_path, f)+'.npy' for f in new_array_names]
    
    #Now, add contrast individually to each array in the dataset
    for i, load_path in enumerate(array_load_paths):
        arr = np.load(load_path)
        noised_array = contrast(arr=arr, sat_val=sat_val, alpha=alpha)
        np.save(file=destination_paths[i], arr=noised_array)


def contrast(arr:np.ndarray, sat_val:float, alpha:float):
    """Scales the image by scaling pixel intensity by some factor.
    
    Parameters
    ----------
        arr : np.ndarray
            The array for which we want to enhance contrast.
        sat_val : float
            Values at this or above are considered saturated- usually this will be drawn from some statistical analysis of the data.
        alpha : float
            Scale factor of the contrast enhancement transformation.

    Returns
    -------
        contrasted_array : np.ndarray
            The array after the constrast operation has been performed.
    """

    #Check to make sure that the value of alpha is valid.
    if alpha < 0:
        raise ValueError(f"Warning: alpha should be nonnegative, but alpha was given as {alpha}")

    #Perform linear transformation on the array.
    contrasted_array = np.multiply(arr, alpha)

    #Clip the array so that all values above saturation are set to 0.
    contrasted_array[contrasted_array > sat_val] = sat_val

    return contrasted_array