import os
import shutil

def main():

    dest_path = './data/processed/horizontal_edge_detector_sets/data_augmentation_method/augmented data'

    # Delete all files in dest_path
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
        os.makedirs(dest_path)  # Recreate the directory
        print(f"Cleaned up {dest_path}")

if __name__ == "__main__":
    main()
