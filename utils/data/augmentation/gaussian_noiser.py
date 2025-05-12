####################################################
# UTILITY FOR ADDING GAUSSIAN NOISE TO THE DATASET #
####################################################

#module imports
import numpy as np
import os

def add_gaussian_noise_to_dataset(source_path:str, dest_path:str, std:float, noising_index:int):
    """Adds gaussian noise to the arrays in a dataset supplied from some path.
    
    Parameters
    ----------
        source_path : str
            The path where the original (unnoised) .npy arrays are stored.
        dest_path : str
            The path where we want to save the arrays after noise has been added.
        std : float
            The standard deviation for the underlying Gaussian distribution that we want to draw noise from.
        noising_index : int
            The iteration number tracking how many times we have added noise at this std to the test images.

    """
     
    #Check to make sure that the std deviation value is valid (nonnegative)
    if std<0:
        raise ValueError(f"Error: standard deviation for Gaussian noising process should be nonnegative, but was passed {std} for this value.")
    
    if std == 0:
        raise ValueError(f"Error: standard deviation should not be 0.")
    
    #Get the individual arrays' paths in the dataset directory
    array_load_paths = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.npy')]
    
    #Modify the names of the array locations to reflect the noising process
    new_array_names = [f[:-4]+f'-gaussian_sigma-{std}-{noising_index}' for f in os.listdir(source_path) if f.endswith('.npy')]
    
    #Also generate the 'save' pathnames in advance
    destination_paths = [os.path.join(dest_path, f)+'.npy' for f in new_array_names]
    
    #Now, add Gaussian noise individually to each array in the dataset
    for i, load_path in enumerate(array_load_paths):
        arr = np.load(load_path)
        noised_array = gaussian_noiser(arr=arr, std=std)
        np.save(file=destination_paths[i], arr=noised_array)
    


def gaussian_noiser(arr:np.ndarray, std:float):
    """Adds gaussian noise to array elementwise.
    
    Parameters
    ----------
        arr : np.ndarray
            The input array who is to be noised by the Gaussian noising process.
        std : float
            Standard deviation for the underlying Gaussian distribution which we are drawing noise from

            
    Returns
    -------
        noised_array : np.ndarray
            The array after the noising process has been performed
    """
    noise = np.random.normal(loc=0, scale=std, size=arr.shape)
    noised_array = arr + noise
    return noised_array

    
