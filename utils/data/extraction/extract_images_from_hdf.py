#############################################
# Extract XRFC TYDNO images from .hdf files #
#############################################
from pyhdf.SD import SD, SDC
import os
import numpy as np

def main():

    #Get path to TDYNO raw shot data- each shot has its own folder inside the data_path directory.
    # i.e. we have directories that look like ./data/raw/tdyno2022_shot_data/{shot directory}
    try:
        rel_raw_data_path = './data/raw/tdyno2022_shot_data/'
        abs_raw_data_path = os.path.abspath(rel_raw_data_path)
        shot_directory_list = os.listdir(abs_raw_data_path)
    except:
        raise FileNotFoundError(f"Error: When looking for raw TDYNO shot data, {abs_raw_data_path} not found. "
                                "Check spelling and raw .hdf file locations.")

    #####################
    # Directories that do not contain the shot images which we want, e.g. PRAD data, etc.
    #####################
    exception_list = ["Neutronics", "PRAD", ".DS_Store", "raw_image_arrays"]
    for excepted_directory in exception_list:
        trial_path = os.path.join(abs_raw_data_path, excepted_directory)
        if os.path.exists(trial_path):
            #Remove excepted directory from shot directories list if the path exists
            shot_directory_list.remove(excepted_directory)
        else:
            print(f"WARNING: Specified excepted directory {trial_path} not found.")

    #######################
    # Extract all necessary .hdf files from the shot directories and store in hdf_file_paths list
    #######################
    hdf_file_paths = []
    for shot_directory in shot_directory_list:
        xrfc_shot_path = os.path.join(abs_raw_data_path, shot_directory, "XRFC3")
        
        #Check to make sure the XRFC3 folder exists...
        if os.path.exists(xrfc_shot_path):
            file_list = os.listdir(xrfc_shot_path)
            for file in file_list:
                if ".hdf" in file:
                    #Get file path if it's an .hdf
                    file_path = os.path.join(xrfc_shot_path, file)
                    hdf_file_paths.append(file_path)
        else:
            print(f"Warning: no XRFC3 folder in {shot_directory}")

    #Create a zipped list of the shot directory with the associated file path
    #   so that we can later save the HDF data arrays using names that correspond to the shot name.
    shot_and_path = list(zip(shot_directory_list,hdf_file_paths))

    ######################
    # Open all .hdf files, first accounting for the fact that they are gzips for some reason
    ######################
    for shot, file_path in shot_and_path:
        try:
            hdf_file = SD(file_path, SDC.READ)
        except:
            raise ValueError(f"Error: {file_path} not a HDF4 file.")

        #Extract the actual data from the image, then store the numpy arrays
        if "Streak_array" in hdf_file.datasets():
            streak_image = hdf_file.select('Streak_array')
            streak_image_data = streak_image[:]

            save_path = os.path.join(abs_raw_data_path, "raw_image_arrays", shot)

            np.save(file=save_path, arr=streak_image_data)


if __name__ == "__main__":
    main()





