#Torch imports
import torch

#Import the homemade dataset class for my own edge images

#Import autoencoder and training function
from training.trainer import Trainer
from testing.tester import Tester
from utils.data_handler import DataHandler

class Pipeline():
    def __init__(self, 
                 model, 
                 train_path, 
                 eval_path, 
                 test_path, 
                 n_epochs, 
                 lr, 
                 optimizer, 
                 batch_size, 
                 model_save_path, 
                 TV_loss_save_path,
                 outputs_save_path,
                 TEST_loss_save_path):
        
        self.train_path = train_path
        self.eval_path = eval_path
        self.test_path = test_path
        self.batch_size = batch_size

        datahandler = DataHandler(
            train_path=self.train_path,
            eval_path=self.eval_path,
            test_path=self.test_path,
            batch_size=self.batch_size
        )

        self.train_loader, self.eval_loader, self.test_loader = datahandler.get_loaders()

        self.model = model
        self.n_epochs = n_epochs
        self.lr = lr
        self.optimizer = optimizer

        self.model_save_path = model_save_path
        self.TV_loss_save_path = TV_loss_save_path
        self.outputs_save_path = outputs_save_path
        self.TEST_loss_save_path = TEST_loss_save_path
        

    def run(self):
        print("Beginning model training ... \n")
        trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            eval_loader=self.eval_loader,
            TV_loss_save_path=self.TV_loss_save_path

        )
        trainer.train(epochs=self.n_epochs)
        self.save_model()

        tester = Tester(
            test_loader=self.test_loader,
            model = self.model,
            outputs_save_path=self.outputs_save_path,
            TEST_loss_save_path=self.TEST_loss_save_path
        )

        tester.test()
        tester.save_outputs()

    def save_model(self):
        print(f"Model trained for {self.n_epochs} epochs. Saving at {self.model_save_path} ...")
        torch.save(self.model.state_dict(), f=self.model_save_path)

