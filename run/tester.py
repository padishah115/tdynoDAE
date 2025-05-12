#import modules
import torch
import numpy as np
import os
import pandas as pd

#type hinting
from torch.utils.data import DataLoader
import torch.nn as nn

class Tester():
    """Class responsible for controlling the training process- calls up the trained model, stores it on the GPU when
    available (otherwise defaults to the CPU), applies the model to the training set, and saves information about the 
    loss on the test set."""

    def __init__(self, test_loader:DataLoader, model:nn.Module, outputs_save_path:str, TEST_loss_save_path:str):
        """Initialization function for the Tester class.

            test_loader : DataLoader
                The iterable DataLoader wrapper around the test set.
            model : nn.Module
                The model to be tested. Should be a derivative of the torch.nn.Module class, and should have
                been saved as a state dictionary during the training stages of the program.
            outputs_save_path : str
                The path to which we will save the outputs of the model after it performs on the test set. This
                way, we can inspect what the model is doing to the test images- is it removing noise?
        """

        # CHECK FOR GPU RESOURCES, DEFAULT TO CPU AS NECESSARY
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # LOADS THE MODEL, SHUNTING TO GPU/CPU
        self.model = model.to(self.device)

        # INITIALIZE TEST DATALOADER
        self.test_loader = test_loader

        # PATHS FOR SAVING MODEL OUTPUTS AND LOSS DATA ABOUT PERFORMANCE ON THE TEST SET
        self.outputs_save_path = outputs_save_path
        self.TEST_loss_save_path = TEST_loss_save_path

        # DICTIONARY TO HOLD THE MODEL'S OUTPUTS BEFORE WE SAVE THEM
        self.outputs = {}

        # KEEP TRACK OF THE TEST LOSS FOR EACH BATCH IN THE TEST DATA
        self.batch_losses = []
        self.mean_batch_loss = 0.0

    def test(self):
        """Function for running the testing process, start-to-finish."""

        # Iterate over batches in the test dataloader, saving loss data in the self.batch_losses list
        for i, items in enumerate(self.test_loader):
            imgs = items[0]
            labels = list(items[1])
            print(f"Testing batch {i+1} of {len(self.test_loader)}")
            self.batch_losses.append(self._test_batch(imgs, labels))
                    
        # Calculate the mean batch loss
        self.mean_batch_loss = np.sum(self.batch_losses) / len(self.batch_losses)
        print(f"TESTING: Mean batch loss = {self.mean_batch_loss}")
        

    def _test_batch(self, imgs:list, labels:list):
            """Test function for the batch. Applies the trained model to the test set, and gathers
            both the model's outputs in the class self.ouputs dictionary, as well as the loss on the batch.
            
            Parameters
            ----------
                imgs : list
                    List containing images in the batch.
                labels : list
                    List containing each image's label (name) in the batch.

            Returns 
            -------
                batch_loss : float
                    The loss on the batch during testing.
            """

            imgs = imgs.to(self.device)
            
            # Checks to see whether our model requires input flattening (i.e. is fully-connected)
            if self.model.flatten:
                batch_size = imgs.shape[0]
                imgs = imgs.view(batch_size, -1)
                output_data = self.model(imgs)
            else:
                output_data = self.model(imgs)
  
            for j, output_datum in enumerate(output_data):
                self.outputs[labels[j]] = output_datum

            # Calculate the batch loss using the model's loss function
            batch_loss = self.model.loss_fn(output_data, imgs).item()

            return batch_loss


    def save_outputs(self):
        """Saves the outputs from the model at a specified path."""

        # Iterate over each output from the model in the test set, and save as a .npy array,
        # giving each file a name corresponding to the initial model input
        for key in self.outputs.keys():
            model_output : torch.Tensor = self.outputs[key] #call up the output from the self.outputs dictionary
            
            if self.model.flatten:
                model_output = model_output.view(2064, -1).detach().numpy() #unflatten if model has FC layer
            else: 
                model_output = model_output.detach().numpy()

            #Save the model's outputs as appropriate
            save_path = os.path.join(self.outputs_save_path, f'from_{key}.npy')
            np.save(file=save_path, arr=model_output)

    def save_loss_data(self):
        """Save data about losses on the test set during testing."""

        df = pd.DataFrame(
            {
                "Mean Batch Loss": [self.mean_batch_loss],
                "Batch Losses" : self.batch_losses
            }
        )
        
        #Save as .csv
        df.to_csv(path_or_buf=self.TEST_loss_save_path, index=False)
        




