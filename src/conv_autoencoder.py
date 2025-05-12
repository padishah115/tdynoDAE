########################################
# TRAIN THE EDGE DETECTION AUTOENCODER #
########################################

#Torch imports
import torch.optim as optim

#Import autoencoder and training function
from models.conv_autoencoder import ConvAutoencoderI
from run.pipeline import Pipeline

def main():

    ########################
    # HERE BE DATA LOADING #
    ########################

    #./data/processed/horizontal_edge_detector_sets/data_augmentation_method/augmented-data/

    #Location of the training and validation sets
    train_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/augmented-data/training-set'
    eval_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/augmented-data/validation-set'
    test_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/augmented-data/test-set' 

    #Model initialisation- here, try the fully-connected autoencoder
    print("Initialising CONV autoencoder model ...\n")
    model = ConvAutoencoderI()
    print("Parameter number",sum([p.numel() for p in model.parameters()]))

    #Initialise hyperparameters of the training loop, and pass to Adam optimizer
    n_epochs = 30
    lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=lr)
    batch_size = 1

    #Location where we seek to save the csv containing information about the loss as a function of epoch number
    TEST_EVAL_loss_save_path = \
        f'./trained_models/First-conv-edge-removal/data-augmentation/loss-statistics/TV_loss_{model.name}-{n_epochs}_epochs.csv'

    TEST_loss_save_path = \
        f'./trained_models/First-conv-edge-removal/data-augmentation/loss-statistics/TEST_loss_{model.name}-{n_epochs}_epochs.csv'
    
    #Location where we will save the state dictionary of the trained model
    model_save_path = \
        f'./trained_models/First-conv-edge-removal/data-augmentation/{model.name}-{n_epochs}_epochs.pt'

    #Model where we will save outputs after trials on the TEST SET
    outputs_save_path = './results/data-augmentation-conv-autoencoder'

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
        TEST_EVAL_loss_save_path=TEST_EVAL_loss_save_path,
        outputs_save_path=outputs_save_path,
        TEST_loss_save_path=TEST_loss_save_path
    )
    pipeline.run()

    print("Training complete, data and model saved.")
    

if __name__ == "__main__":
    main()