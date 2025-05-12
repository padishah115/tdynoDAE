################################################### 
# FILE CONTAINING GENERIC TRAINING LOOP FUNCTIONS #
###################################################

#Module imports- pytorch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

#General module imports for data handling
import pandas as pd
import os


class Trainer():
    """Class which encapsulates the full training process during program run- calls up the model and optimizer,
    and performs optimization over the specified number of epochs."""

    def __init__(self, 
                 model:nn.Module, 
                 optimizer:optim.Optimizer, 
                 train_loader:DataLoader, 
                 eval_loader:DataLoader, 
                 TRAIN_EVAL_loss_save_path:str
                 ):

        # CHECKS TO SEE WHETHER GPU IS AVAILABLE AND ADJUSTS DEVICE AS APPROPRIATE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        # Initialize optimizer and dataloaders for the test and validation sets.
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # Location for storing loss statistics on the training and validation sets as a function of epoch number.
        self.TRAIN_EVAL_loss_save_path = TRAIN_EVAL_loss_save_path

        # Lists for tracking loss data.
        self.epochs = []
        self.training_losses = []
        self.validation_losses = []

    def train(self, epochs:int):
        """Runs the training process on the model.
        
        Parameters
        ----------
            epochs : int
                Number of epochs over which training is performed.
        """

        #Run the training process over the specified number of epochs.
        for epoch in range(1, epochs+1):
            print(f"Entering epoch {epoch} ...\n") 
            
            mean_train_batch_loss = self._train_epoch() #get mean training batch loss
            mean_val_batch_loss = self._validate() #get mean validation batch loss
            self.epochs.append(epoch) # store epoch number for loss data saving
            
            # store mean training/validation batch losses for each epoch
            self.training_losses.append(mean_train_batch_loss)
            self.validation_losses.append(mean_val_batch_loss)
            print(f'TRAINING: Mean batch loss at epoch {epoch} = {mean_train_batch_loss}\n')
            print(f'VALIDATION: Mean batch loss at epoch {epoch} = {mean_val_batch_loss}\n')



    def _train_epoch(self)->float:
        """Helper function encoding iteration over each batch for a given epoch.
        
        Returns
        -------
            mean_batch_loss_TRAIN : float
                The mean loss across all batches in the training set as a float.
        """

        loss_train = 0.0
        #Iterate over image batches in training dataloader
        
        for i, items in enumerate(self.train_loader):
            print(f"Training Batch no. : {i+1}\n")
            
            # 0th entry in items object is the batch data, 1th entry are the labels
            imgs = items[0]
            imgs = imgs.to(self.device)

            #Flattens input data if the model is fully-connected
            if self.model.flatten:
                #Reshape the input images so that they are flat
                batch_size = imgs.shape[0]
                inputs = imgs.view(batch_size, -1)
            else:
                inputs = imgs

            #   FORWARDS PASS
            # Apply model to training bass, calculate losses.
            outputs = self.model(inputs) 
            loss = self.model.loss_fn(inputs, outputs)

            #FREE MEMORY
            del imgs, outputs
            torch.cuda.empty_cache()

            #   BACKWARDS PASS
            self.optimizer.zero_grad() #prevent gradient accumulation
            loss.backward() #call backpropagation on loss function
            self.optimizer.step() #update model parameters
            loss_train += loss.item() #accumulate batch loss
        
        #Compute the mean loss over batches
        mean_batch_loss_TRAIN = loss_train / len(self.train_loader)

        # RETURN the mean batch loss.
        return mean_batch_loss_TRAIN

    def _validate(self)->float:
        """Calculates mean loss across batches in the validation set, allowing us to monitor the extent of the model's
        variance (overfitting) from epoch to epoch.
        
        Returns
        -------
            mean_batch
        """

        eval_loss = 0.0
        
        #Perform predictions over members of validation set
        with torch.no_grad(): #no backpropagation performed during validation.
            for i, items in enumerate(self.eval_loader):
                imgs = items[0] #0th entry in batch is the data itself, 1th entry is the labels.
                print(f"Validation batch no: {i+1}\n")
                imgs = imgs.to(self.device) #load images onto GPU if available

                #   FORWARDS PASS
                #Get model outputs for the batch and calculate loss
                #Flattens input images if model is fully-connected
                if self.model.flatten:
                    batch_size = imgs.shape[0]
                    inputs = imgs.view(batch_size, -1)
                else:
                    inputs = imgs

                # Apply model to validation data, calculate loss on the validation batch.
                outputs = self.model(inputs)
                eval_loss += self.model.loss_fn(inputs, outputs).item() #use .item() to avoid gradient calculations
                
                #Free up memory on the GPU (CPU)
                del imgs, outputs
                torch.cuda.empty_cache()
        
        # Calculate the mean validation loss across all batches, and return.
        mean_batch_loss_EVAL = eval_loss / len(self.eval_loader)
        return mean_batch_loss_EVAL
    
    def save_losses(self):
        """Save mean loss data across training and validation batches as a function of epoch number. This data
        will be stored in a three-column .csv file, with column headers 
        ["Epochs", "Mean Batch Loss (Training)", "Mean Batch Loss (Validation)"]."""

        #Initialize dataframe, and store epoch and loss data.
        df = pd.DataFrame(
        data={
            "Epochs" : self.epochs,
            "Mean Batch Loss (Training)": self.training_losses,
            "Mean Batch Loss (Validation)": self.validation_losses
        }
    )

        if not self.TRAIN_EVAL_loss_save_path.endswith('.csv'):
            os.path.join(self.TRAIN_EVAL_loss_save_path, '.csv')

        # Save the CSV, without an index (this is useless and annoying, and the epoch number will take care)
        # of this anyway. 
        print("Saving loss data for training and validation ...\n")
        df.to_csv(self.TRAIN_EVAL_loss_save_path, index=False)
