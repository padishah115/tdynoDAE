import sys
sys.path.append(".")

from utils.data.datasets.create_random_datasets import create_3_random_datasets

def main():

    create_3_random_datasets(
        data_path="/Users/hayden/Desktop/TDYNO/tdynoDAE-data/dummyset",
        dest_path="/Users/hayden/Desktop/TDYNO/tdynoDAE-data/dummyset",
        train_frac=0.70,
        eval_frac=0.15,
        test_frac=0.15
    )

    

if __name__ == "__main__":
    main()