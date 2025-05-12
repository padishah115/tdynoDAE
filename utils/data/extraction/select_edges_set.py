#####################################################################################################################################
# Selects the IDENTITY images for 1044{69, 70, 73, 75, 77} so that we can use these as the test set in our data augmentation method #
#####################################################################################################################################

#Module imports
import os
import shutil

def main():

    #Path to the directory where we have the background-subtracted images stored as arrays
    raw_corrected_arrays_path = './data/raw/tdyno2022_shot_data/corrected_image_arrays'
    
    destination_path = './data/processed/horizontal_edge_detector_sets/identity_method/test set'
    
    #Search the directory for the images that we want and get their paths
    selected_images = ['104469.npy', '104470.npy', '104473.npy', '104475.npy', '104477.npy']
    selected_images_paths = []
    list_dir = os.listdir(raw_corrected_arrays_path)
    print("Directory list:", list_dir)
    
    for entry in list_dir:
        if entry in selected_images:
            array_path = os.path.join(raw_corrected_arrays_path, entry)
            selected_images_paths.append(array_path)

            array_destination = os.path.join(destination_path, entry)

            shutil.copy(array_path, array_destination)


if __name__ == "__main__":
    main()
