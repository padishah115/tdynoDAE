# IMPORT DATALOADER FROM PYTORCH, AND MY OWN CUSTOM NPYDATASET CLASS
from torch.utils.data import DataLoader
from utils.data.datasets.npy_dataset import npyDataset

class DataHandler():
    """Class responsible for returning us the datasets and dataloaders, created from .npy arrays."""

    def __init__(self, train_path:str, eval_path:str, test_path:str, batch_size:int):
        """Initialization function for the DataHandler class.
        
        Parameters
        ----------
            train_path : str
                Path to the training data directory, where the data is stored as .npy files.
            eval_path : str
                Path to the validation data directory, where the data is stored as .npy files.
            test_path : str
                Path to the test data directory, where the data is stored as .npy files.
            batch_size : int
                Size of the batch which we will use during SGD optimization.
        """

        # CREATE CUSTOM DATASETS FROM DATA AT SPECIFIED TEST, VALIDATION, AND TRAINING DATA DIRECTORIES 
        self.train_set = npyDataset(arrays_path=train_path)
        self.eval_set = npyDataset(arrays_path=eval_path)
        self.test_set = npyDataset(arrays_path=test_path)
        
        # WRAP THE DATASETS IN TORCH DATALOADERS
        # NOTE- ONLY SHUFFLE THE TRAINING SET DATALOADER
        self.train_loader = DataLoader(dataset=self.train_set, shuffle=True, batch_size=batch_size)
        self.eval_loader = DataLoader(dataset=self.eval_set, shuffle=False, batch_size=batch_size)
        self.test_loader = DataLoader(dataset=self.test_set, shuffle=False, batch_size=batch_size)

    def get_sets(self)->tuple[npyDataset, npyDataset, npyDataset]:
        """Returns a training set, validation set, and test set, created using the npyDataset class
        which have generated data from specified locations.
        
        Returns
        -------
            self.train_set : npyDataset
                The training set in dataset form (NOT dataloader form).
            self.eval_set : npyDataset
                The validation set in dataset form (NOT dataloader form).
            self.test_set : npyDataset
                The test set in dataset form (NOT dataloader form).
        """

        return self.train_set, self.eval_set, self.test_set
    
    def get_loaders(self)->tuple[DataLoader, DataLoader, DataLoader]:
        """Returns the training, validation, and test dataloaders in order to allow iteration
        during optimization.
        
        Returns
        -------
            self.train_loader
                The iterable training data DataLoader.
            self.eval_loader
                The iterable training data DataLoader.
            self.test_loader
                The iterable training data DataLoader.
        """

        return self.train_loader, self.eval_loader, self.test_loader