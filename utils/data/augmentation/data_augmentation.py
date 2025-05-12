#########################################################################################
# AUGMENTED DATASET CREATOR FOR OUR DATA-AUGMENTED EDGE DETECTION METHOD                #
# Adds SnP and Gaussian noise to test dataset, and then saves the noised images for us. #
#########################################################################################

#Module imports
import numpy as np
import pandas as pd
from typing import List
import sys
import os

# Add the src directory to sys.path to allow utils imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

#import my own noising scripts
from utils.data.augmentation.snp_noiser import add_snp_noise_to_dataset
from utils.data.augmentation.gaussian_noiser import add_gaussian_noise_to_dataset
from utils.data.augmentation.contrast import add_contrast_to_dataset

def snp(source_path:str, dest_path:str, sat_val:float, bernoulli_params:List[float], noisy_images_per_param:int):
    """Wrapper function for adding all of our desired snp noise."""

    for bernoulli_p in bernoulli_params:
        for noising_index in range(noisy_images_per_param):
            add_snp_noise_to_dataset(
                source_path=source_path,
                dest_path=dest_path,
                sat_val=sat_val,
                bernoulli_p=bernoulli_p,
                noising_index=noising_index
            )

    #No need for returns as the add_snp_noise function saves the arrays for us.


def gaussian(source_path:str, dest_path:str, stds:List[float], noisy_images_per_param:int):
    """Wrapper function for adding all of our desired Gaussian noise."""
    for std in stds:
        for noising_index in range(noisy_images_per_param):
            add_gaussian_noise_to_dataset(
                source_path=source_path,
                dest_path=dest_path,
                std=std,
                noising_index=noising_index
            )

def contrast(source_path:str, dest_path:str, sat_val:float, alphas:List[float]):
    """Wrapper function for adding all of our desired contrast."""
    for alpha in alphas:
        add_contrast_to_dataset(
            source_path=source_path,
            dest_path=dest_path,
            sat_val=sat_val,
            alpha=alpha,
        )


def main():

    #Source path to the original images, which I hand-selected from real data due to their sharp edges. This contains 6 "base classes"
    source_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/sorted data/test set'
    
    #Dump all of the noised images in the "noised data" folder before randomly sorting into training and val
    dest_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/augmented data'

    #####################################################################################################
    # CALL UP OUR STATISTICAL ANALYSIS OF THE DATASET IN ORDER TO CALCULATE OUR AUGMENTATION PARAMETERS #
    #####################################################################################################

    #Path to the statistics of the test dataset
    stats_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/sorted data/test_set_stats.csv'

    #Check to make sure that the stats path points to a 
    if not stats_path.endswith('.csv'):
        raise ValueError(f"Error: stats_path should point to a .csv file, but supplied path {stats_path} does not.")

    #Read off the dataset's statistics from the stats_path- particularly the MAX pixel intensity, which we will use as the saturation value.
    dataset_stats = pd.read_csv(stats_path)
    dataset_pmax = dataset_stats["pmax"].iloc[0] #maximum pixel intensity in the dataset
    
    ################################
    # DATA AUGMENTATION PARAMETERS #
    ################################

    print("Initialising noise parameters ... \n")

    #We are going to generate many SnP'd images with varying levels of SnP noise
    #   ranging from phi = 0.01 to 0.25
    bernoulli_params = np.arange(start=0.01, stop=0.26, step=0.01)
    
    #Standard deviations ranging from 0.5% to 12.5% of the maximum pixel intensity.
    stds = np.multiply(np.arange(0.005, 0.130, 0.005), dataset_pmax)

    #Alpha values for contrasted images.
    alphas = np.linspace(1.004, 2, 250)
    
    #We want to generate 10 images per base class image per parameter (for SnP: Bernoulli param ; for Gauss: std ; for contrast: alpha.
    noisy_images_per_param = 10

    ##############################
    # CALL THE DATA AUGMENTATION #
    ##############################

    print("Adding SnP noise to dataset ... \n")

    snp(
        source_path=source_path,
        dest_path=dest_path,
        sat_val=dataset_pmax,
        bernoulli_params=bernoulli_params,
        noisy_images_per_param=noisy_images_per_param
    )

    print("Adding Gaussian noise to dataset ... \n")

    gaussian(
        source_path=source_path,
        dest_path=dest_path,
        stds=stds,
        noisy_images_per_param=noisy_images_per_param    
    )

    print("Increasing image contrast of dataset ... \n")

    contrast(
        source_path=source_path,
        dest_path=dest_path,
        sat_val=dataset_pmax,
        alphas=alphas,
    )

if __name__ == "__main__":
    main()
