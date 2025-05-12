########################################
# TRAIN THE EDGE DETECTION AUTOENCODER #
########################################

#Torch imports
import torch.optim as optim

#Import autoencoder and training function
from models.fc_autoencoder import FcAutoencoder
from run.pipeline import Pipeline

def main():

    ########################
    # HERE BE DATA LOADING #
    ########################

    #Location of the training and validation sets
    train_path = './data/processed/horizontal_edge_detector_sets/multiple_filters_method/untrimmed/training set'
    eval_path = './data/processed/horizontal_edge_detector_sets/multiple_filters_method/untrimmed/validation set'
    test_path = './data/processed/horizontal_edge_detector_sets/multiple_filters_method/untrimmed/test set' 

    #Model initialisation- here, try the fully-connected autoencoder
    print("Initialising FC autoencoder model ...\n")
    model = FcAutoencoder()

    #Initialise hyperparameters of the training loop, and pass to Adam optimizer
    n_epochs = 1
    lr = 1e-2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size = 2

    #Location where we seek to save the csv containing information about the TRAINING AND VALIDATION loss as a function of epoch number
    TV_loss_save_path = f'./trained-models/test/TV_loss_{model.name}-{n_epochs}_epochs.csv'

    TEST_loss_save_path = f'./trained-models/test/TEST_loss_{model.name}-{n_epochs}_epochs.csv'
    
    #Location where we will save the state dictionary of the trained model
    model_save_path = f'./trained-models/test/{model.name}-{n_epochs}_epochs.pt'

    #Model where we will save outputs after trials on the TEST SET
    outputs_save_path = './results/test'

    #Run training loop
    print("Beginning model training ...\n")
    
    pipeline = Pipeline(
        model=model,
        train_path=train_path,
        eval_path=eval_path,
        test_path = test_path,
        n_epochs=n_epochs,
        lr=lr,
        optimizer=optimizer,
        batch_size=batch_size,
        model_save_path=model_save_path,
        TV_loss_save_path=TV_loss_save_path,
        outputs_save_path=outputs_save_path,
        TEST_loss_save_path = TEST_loss_save_path
    )
    pipeline.run()

    print("Training and testing complete, data and model saved.")
    

if __name__ == "__main__":
    main()
