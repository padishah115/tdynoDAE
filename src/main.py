import sys
sys.path.append(".")

# from utils.data.imageprocessing.background_correction import correct_arrays
# from utils.data.imageprocessing.unsqueeze_arrays import unsqueeze_arrays
# from utils.data.imageprocessing.trim_images import trim_dataset
# from utils.data.datasets.create_random_datasets import create_3_random_datasets

from train_conv_autoencoder import main as train_conv_auto_encoder

def main():

    # source_path = "/Users/hayden/Desktop/TDYNO/tdynoDAE-data/dummyset"

    # unsqueeze_arrays(
    #     data_path=source_path
    # )



    # create_3_random_datasets(
    #     data_path="/Users/hayden/Desktop/TDYNO/tdynoDAE-data/dummyset",
    #     dest_path="/Users/hayden/Desktop/TDYNO/tdynoDAE-data/dummyset",
    #     train_frac=0.70,
    #     eval_frac=0.15,
    #     test_frac=0.15
    # )

    train_conv_auto_encoder()

    

if __name__ == "__main__":
    main()