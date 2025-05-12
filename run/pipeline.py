# IMPORT TORCH MODULES FOR TYPE HINTING
import torch
import torch.nn as nn
import torch.optim as optim

# Import autoencoder training and testing functions
from run.trainer import Trainer
from run.tester import Tester

# IMPORT DATAHANDLER CLASS (homemade)
from utils.data.datasets.data_handler import DataHandler



class Pipeline():
    """Class for encapsulating the entire training and testing process."""
    
    def __init__(self, 
                 model:nn.Module, 
                 train_path:str, 
                 eval_path:str, 
                 test_path:str, 
                 n_epochs:int, 
                 lr:float, 
                 optimizer:optim.Optimizer, 
                 batch_size:int, 
                 model_save_path:str, 
                 TRAIN_EVAL_loss_save_path:str,
                 outputs_save_path:str,
                 TEST_loss_save_path:str):
        """Initialization function for the Pipeline class.
        
        Parameters
        ----------
            model : nn.Module
                The model which we want to train. This should be a derivative of the nn.Module class.
            train_path : str
                The path to the training dataset.
            eval_path : str
                The path to the validation dataset.
            test_path : str
                The path to the test dataset.
            n_epochs : int
                The number of epochs for which we want to train the model.
            lr : float
                The learning rate for gradient descent/optimization.
            optimizer : optim.Optimizer
                The optimizer (e.g. ADAM or AdaGrad) which we want to use to adjust model parameters.
            batch_size : int
                The size of the batch used in stochastic gradient descent (SGD). Recall that we only want to perform
                gradient descent over batches, as processing the entire dataset simultaneously is an incredibly slow 
                and ineffectual process.
            model_save_path : str
                The path to where we would like to store the model's state dictionary after training and testing.
            TRAIN_EVAL_loss_save_path : str
                The path to where we will store loss information about how well the model is performing on the training
                and validation sets at each epoch.
            outputs_save_path : str
                The path to which we would like to save the model's (hopefully) denoised ouputs. These outputs are produced
                by the model acting on images in the test dataset.
            TEST_loss_save_path : str
                The path where we would like to store information about loss on the test dataset.
        """
        
        # INITIALIZE PATHS TO TRAINING, VALIDATION, AND TEST DATASETS
        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path

        # SET THE BATCH SIZE FOR TRAINING
        self.batch_size = batch_size

        # INITIALIZE THE DATAHANDLER INSTANCE
        datahandler = DataHandler(
            train_path=self.train_path,
            eval_path=self.eval_path,
            test_path=self.test_path,
            batch_size=self.batch_size
        )
        # get the dataloaders for the training, validation, and test data from the datahandler class
        self.train_loader, self.eval_loader, self.test_loader = datahandler.get_loaders()

        # Initialize information about the model and optimization
        self.model = model # model as nn.Module
        self.n_epochs = n_epochs # number of epochs over which training will occur
        self.lr = lr # learning rate as a float
        self.optimizer = optimizer # optimizer to be used on the model during gradient descent.

        # Paths for saving data about the training and testing processes.
        self.model_save_path = model_save_path # model state dict
        self.TRAIN_EVAL_loss_save_path = TRAIN_EVAL_loss_save_path # loss stats for training and validation
        self.outputs_save_path = outputs_save_path # model outputs from test data
        self.TEST_loss_save_path = TEST_loss_save_path # path where we'll save information about loss on the test set
        

    def run(self):
        """Runs the pipeline start-to-finish- loads data from the appropriate directory, trains the model
        over the specified number of epochs, tests the model on the test set, saves loss statistics, and saves
        the model's outputs after action on the test set"""

        print("Beginning model training ... \n")
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            eval_loader=self.eval_loader,
            TRAIN_EVAL_loss_save_path=self.TRAIN_EVAL_loss_save_path

        )
        #Train the model on the training set
        trainer.train(epochs=self.n_epochs)
        #Save loss statistics for the training and validation
        trainer.save_losses()
        #save the model's state dictionary
        self._save_model()

        #Use the tester class to handle testing of the saved model
        tester = Tester(
            test_loader=self.test_loader,
            model = self.model,
            outputs_save_path=self.outputs_save_path,
            TEST_loss_save_path=self.TEST_loss_save_path
        )
        #Load the model, test it on the test set
        tester.test()
        #Save the model's outputs, i.e. what the model produces in the output layer
        tester.save_outputs()

        #Save information about losses on the test set
        tester.save_loss_data()


    def _save_model(self):
        """Saves the model's state dictionary at a specified path."""

        print(f"Model trained for {self.n_epochs} epochs. Saving at {self.model_save_path} ...")
        torch.save(self.model.state_dict(), f=self.model_save_path)

