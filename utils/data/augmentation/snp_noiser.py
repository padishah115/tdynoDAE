####################################################################
# UTILITY FOR ARTIFICIALLY ADDING SALT AND PEPPER NOISE TO DATASET #
####################################################################

#Module imports
import numpy as np
import pandas as pd
import os

#Need to import dataset parameters from some .csv so that we know what to set the maximum pixel value to.

def add_snp_noise_to_dataset(source_path:str, dest_path:str, sat_val:float, bernoulli_p:float, noising_index:int):
    """Adds salt and pepper noise to a full dataset at some location, and then saves the noised dataset somewhere.
    Note: so far we are assuming that both types of saturation occur with equal probability.
    
    Parameters
    ----------
        source_path : str
            The path where the original (unnoised) .npy arrays are stored.
        dest_path : str
            The path where we want to save the arrays after noise has been added.
        sat_val : float
            The value of pixel intensity at which we are considering the pixel to be saturated. Typically, this is the max measured intensity.
        bernoulli_p : float
            The Bernoulli parameter of the SnP noising process, determining the likelihood of each pixel being saturated/dead or not
        noising_index : int
            The number of times thus far that we have added SnP noise with this bernoulli parameter.
    """

    
    #Check to make sure that the bernoulli parameter passed to the function is a valid probability
    if bernoulli_p<0 or bernoulli_p>1:
        raise ValueError(f"Warning: Bernoulli parameter should be between 0 and 1, but was given {bernoulli_p} as parameter value.")    
    

    #Get the individual arrays' paths in the dataset directory
    array_load_paths = [os.path.join(source_path, f) for f in os.listdir(source_path) if f.endswith('.npy')]
    
    #Modify the names of the array locations to reflect the noising process
    new_array_names = [f[:-4]+f'-snp_phi-{bernoulli_p}-{noising_index}' for f in os.listdir(source_path) if f.endswith('.npy')]
    
    #Also generate the 'save' pathnames in advance
    destination_paths = [os.path.join(dest_path, f)+'.npy' for f in new_array_names]
    
    #Now, add SNP noise individually to each array in the dataset
    for i, load_path in enumerate(array_load_paths):
        arr = np.load(load_path)
        noised_array = snp_noiser(arr=arr, sat_val=sat_val, bernoulli_p=bernoulli_p)
        np.save(file=destination_paths[i], arr=noised_array)


def snp_noiser(arr:np.ndarray, sat_val:float, bernoulli_p:float):
    """Adds salt and pepper noise to some input array.
    
    Parameters
    ----------
        arr : array
            The input array to whom we want to add the salt and pepper noise
        sat_val : float
            Pixel intensity associated with salted saturation (ensemble maximum for the training set)
        bernoulli_p : float
            The Bernoulli parameter of the SnP noising process, determining the likelihood of each pixel being saturated or not

    Returns
    -------
        noised_arr: array
            The array after the noising process has been performed.

    """

    #Check to make sure that the array is a single-channel image.
    if len(arr.shape) != 3:
        raise ValueError(f"Error: expected array of shape (1, H, W), got array of shape {arr.shape}")
    
    #Check to make sure that the image has one channel, i.e. is a grayscale image.
    if arr.shape[0] != 1:
        raise ValueError(f"Error: expected single-channel grayscale image, got image with {arr.shape[0]} channels.")
    
    #Make sure that the bernoulli parameter is a valid probability, i.e. a non-negative value between 0 and 1
    if bernoulli_p<0 or bernoulli_p>1:
        raise ValueError(f"Error: Bernoulli parameter should be between 0 and 1, but was given {bernoulli_p} as parameter value.")

    #Boolean matrices
    mask = np.random.rand(*arr.shape) < bernoulli_p #determines which pixels are corrupted
    salt = np.random.rand(*arr.shape) > 0.5 #mask & salt will determine which corrupted pixels are salt or pepper

    #Now generate the noised array and return
    noised_arr = np.where(mask&salt, sat_val, np.where(mask, 0, arr))

    return noised_arr
    

    