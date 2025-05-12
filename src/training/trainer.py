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
    def __init__(self, model:nn.Module, optimizer, train_loader, eval_loader, TV_loss_save_path):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.TV_loss_save_path = TV_loss_save_path

        self.epochs = []
        self.training_losses = []
        self.validation_losses = []

    def train(self, epochs:int):
        for epoch in range(1, epochs+1):
            print(f"Entering epoch {epoch} ...\n") 
            train_batch_loss = self.train_epoch()
            val_batch_loss = self._validate()
            self.epochs.append(epoch)
            self.training_losses.append(train_batch_loss)
            self.validation_losses.append(val_batch_loss)
            print(f'TRAINING: Mean batch loss at epoch {epoch} = {train_batch_loss}\n')
            print(f'VALIDATION: Mean batch loss at epoch {epoch} = {val_batch_loss}\n')
        
        self.save_losses()



    def train_epoch(self):
        loss_train = 0.0
        for i, items in enumerate(self.train_loader):
            print(f"Training Batch no. : {i+1}\n")
            imgs = items[0]
            imgs = imgs.to(self.device)

            #Clause that can deal with fully-connected models.
            if self.model.flatten:
                #Reshape the input images so that they are flat
                batch_size = imgs.shape[0]
                inputs = imgs.view(batch_size, -1)
            else:
                inputs = imgs

            #   FORWARDS PASS
            outputs = self.model(inputs)
            loss = self.model.loss_fn(inputs, outputs)

            #FREE MEMORY
            del imgs, outputs
            torch.cuda.empty_cache()

            #   BACKWARDS PASS
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_train += loss.item()
        #Compute the mean loss over batches
        mean_batch_loss_TRAIN = loss_train / len(self.train_loader)
        return mean_batch_loss_TRAIN

    def _validate(self):
        eval_loss = 0.0
        #Perform predictions over members of validation set
        with torch.no_grad():
            for i, items in enumerate(self.eval_loader):
                imgs = items[0]
                print(f"Validation batch no: {i+1}\n")
                imgs = imgs.to(self.device)

                #   FORWARDS PASS
                #Get model outputs for the batch and calculate loss
                if self.model.flatten:
                    batch_size = imgs.shape[0]
                    inputs = imgs.view(batch_size, -1)
                else:
                    inputs = imgs

                outputs = self.model(inputs)
                eval_loss += self.model.loss_fn(inputs, outputs).item() #use .item() to avoid gradient calculations
                del imgs, outputs
        
        mean_batch_loss_EVAL = eval_loss / len(self.eval_loader)
        return mean_batch_loss_EVAL
    
    def save_losses(self):
        df = pd.DataFrame(
        data={
            "Epochs" : self.epochs,
            "Mean Batch Loss (Training)": self.training_losses,
            "Mean Batch Loss (Validation)": self.validation_losses
        }
    )

        if not self.TV_loss_save_path.endswith('.csv'):
            os.path.join(self.TV_loss_save_path, '.csv')

        #Save the CSV
        print("Saving loss data for training and validation ...\n")
        df.to_csv(self.TV_loss_save_path, index=False)
