###########################################
# CUSTOM DATASET CLASS FOR LOADING IMAGES #
###########################################

#Module imports
import os
import torch
import numpy as np

#Import dataset base class
from torch.utils.data import Dataset, DataLoader

#Import transforms library
import torchvision.transforms as transforms

class npyDataset(Dataset):
    """Custom dataset class for loading user-provided .npy files."""
    
    def __init__(self, arrays_path, transform=torch.from_numpy):
        """Initialization function for the npyDataset class.
        
        Parameters
        ----------
            arrays_path : str
                Path containing the .npy arrays which we want to group into a dataset.
            transform : callable
                Transformation function to be applied to the loaded dataset.
        """

        # Call initialization function on parent Dataset class.
        super().__init__()

        #Path to the raw .npy arrays.
        self.arrays_path = arrays_path
        
        #Initialize the transform callable as a class attribute
        self.transform = transform
        
        # create a list of the .npy filenames contained at the target directory.
        self.array_file_list = [f for f in os.listdir(self.arrays_path) if f.endswith('.npy')]

    def __len__(self):
        """Length dunder method."""
        return len(self.array_file_list)
    
    def __getitem__(self, index):
        """Returns the item and its name"""

        # Get the path (location) of the numpy array
        item_path = os.path.join(self.arrays_path, self.array_file_list[index])
        # Load the numpy array from the path
        sample = np.load(item_path)

        # Transform if a transform is provided
        sample = self.transform(sample)

        sample = sample.to(torch.float32)
        
        # Name of the file- important for tracking which outputs come from which input during testing
        label = os.path.splitext(self.array_file_list[index])[0]
        
        # Return sample first, then label
        return sample, label
