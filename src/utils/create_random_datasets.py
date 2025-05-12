##############################################################################################
# SCRIPT FOR RANDOMLY GENERATING TRAINING, VALIDATION, AND TEST SETS FROM AUGMENTED DATASETS #
##############################################################################################

#Module imports
import os
import shutil
import numpy as np

def create_3_random_datasets(data_path:str, dest_path:str, train_frac:float, eval_frac:float, test_frac:float):
    """Function which creates training, validation, and test datasets from an overall collection of data
    by using pseudorandom sampling.

    Parameters
    ----------
        data_path : str
            Path to the location where the original (unseparated) data is stored.
        dest_path : str
            Path to parent directory containing the "test", "validation", and "training" set subdirectories
        train_frac : float
            Fraction (as a decimal) of data which we want to reserve for the training set.
        eval_frac : float
            Fraction (as a decimal) of data which we want to reserve for the validation set.
        test_frac : float
            Fraction (as a decimal) of data which we want to reserve for the test set.
    
    """

    #Give path to the data before it has been sorted into three data sets
    #   Also gets us some information about the data (i.e. how much of it we have)
    arrays_list = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    dataset_size = len(arrays_list)

    train_set_length = int(np.round(train_frac * dataset_size))
    eval_set_length = int(np.round(eval_frac * dataset_size))
    test_set_length = int(np.round(test_frac * dataset_size))
    total_set_length = train_set_length + eval_set_length + test_set_length

    print(f"Training set length : {train_set_length} \nValidation set length: {eval_set_length} \nTest set length: {test_set_length}")
    print(f"Total (true) set length: {dataset_size}")

    if total_set_length != dataset_size:
        raise ValueError("\nWarning: the total dataset size does not match the sum of the the individual sets.")
    
    sets = ['training', 'validation', 'test']
    for set in sets:
        subdir_path = os.path.join(dest_path, f'{set} set')
        os.makedirs(subdir_path, exist_ok=True)

    training_examples = np.random.choice(arrays_list, size=train_set_length, replace=False).tolist()
    for training_example in training_examples:
        arrays_list.remove(training_example)
        old_example_path = os.path.join(data_path, training_example)
        new_example_path = os.path.join(dest_path, 'training set', training_example)
        shutil.copy(old_example_path, new_example_path)

    eval_examples = np.random.choice(arrays_list, size=eval_set_length, replace=False).tolist()
    for eval_example in eval_examples:
        arrays_list.remove(eval_example)
        old_example_path = os.path.join(data_path, eval_example)
        new_example_path = os.path.join(dest_path, 'validation set', eval_example)
        shutil.copy(old_example_path, new_example_path)

    test_examples = arrays_list
    for test_example in test_examples:
        old_example_path = os.path.join(data_path, test_example)
        new_example_path = os.path.join(dest_path, 'test set', test_example)
        shutil.copy(old_example_path, new_example_path)

    print(f'Training examples: {training_examples}\
          \nValidation examples: {eval_examples}\
          \nTest examples: {test_examples}')
    




