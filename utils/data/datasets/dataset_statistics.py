#########################################################################
# GIVES US STATISTICS ABOUT SOME DESIRED DATASET AT SOME SPECIFIED PATH #
#########################################################################

#Module imports
import os
import numpy as np
import pandas as pd
from typing import Tuple

def get_mean(arr_list:list):
    """Gets mean from a list of arrays passed as a parameter.
    
    Parameters
    ----------
        arr_list : list
            List of arrays whose mean we want to calculate

    Returns
    -------
        mean : float
            Mean pixel intensity across all elements of an array
    """
    
    total = 0.
    pixel_no = 0.
    for arr in arr_list:
        total += np.sum(arr)
        pixel_no += np.size(arr)

    mean = total / pixel_no

    return float(mean)




def get_med(arr_list:list):
    """Returns median value from a list of arrays"""

    no_list = []
    for arr in arr_list:
        no_list.extend(arr.flatten().tolist())

    median = np.median(no_list)

    return float(median)





def get_max(arr_list:list):
    """Gets maximum pixel intensity from a list of arrays"""

    max = 0.

    for arr in arr_list:
        arr_max = np.max(arr)
        if arr_max > max:
            max = arr_max

    return float(max)



def get_min(arr_list:list):
    """Gets minimum pixel intensity from a list of arrays"""

    min = get_max(arr_list=arr_list)

    for arr in arr_list:
        arr_min = np.min(arr)
        if arr_min < min:
            min = arr_min

    return float(min)




def get_stats(dataset_path:str)->Tuple:
    """
    
    Parameters
    ----------
        dataset_path : str
            The path to the dataset whose statistics we are interested in

    Returns
    -------
        pmean : float
            Mean pixel intensity of the dataset
        pmed : float
            Median pixel intensity of the dataset
        pmax : float
            Max pixel intensity of the dataset
        pmin : float

    """

    #Retrive the names and thus paths of all the files in the target directory with '.npy' file extensions
    array_path_list = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    array_list = [np.load(array_path_list[i]) for i in range(len(array_path_list))]

    pmean = get_mean(arr_list=array_list)
    pmed = get_med(arr_list=array_list)
    pmax = get_max(arr_list=array_list)
    pmin = get_min(arr_list=array_list)

    return pmean, pmed, pmax, pmin

if __name__ == "__main__":
    path = "./data/processed/horizontal_edge_detector_sets/data_augmentation_method/test set/"
    pmean, pmed, pmax, pmin = get_stats(dataset_path=path)

    stat_dict = {"pmean":[pmean], "pmed":[pmed], "pmax": [pmax], "pmin": [pmin]}
    df = pd.DataFrame(stat_dict)
    
    save_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/test_set_stats.csv'
    df.to_csv(save_path)