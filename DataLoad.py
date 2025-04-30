import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(20230226)  # Fix random seed
torch.backends.cudnn.deterministic = True  # Fix GPU computation method
amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
def getSequenceData(direction: str):
    # Load data from target path
    data, label = [], []
    max_length = 0
    min_length = 8000

    with open(direction) as f:  # Read file
        for each in f:  # Loop 1: Each line in the file
            each = each.strip()  # Remove spaces at the beginning and end of the string
            each = each.upper()  # Convert lowercase to uppercase
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                max_length = max(max_length, len(each.split('\n')[0]))  # Maximum sequence length
                min_length = min(min_length, len(each.split('\n')[0]))  # Minimum sequence length
                data.append(each)
    return np.array(data), np.array(label), max_length, min_length
def PadEncode(data, label, max_len: int = 50):
    # Sequence encoding
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        if len(data[i]) > max_len:  # Remove sequences with length greater than 50
            continue
        element, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:  # Remove sequences containing non-natural amino acids
                sign = 1
                break
            index = amino_acids.index(j)  # Get letter index
            element.append(index)  # Replace letter with number
            sign = 0

        if length <= max_len and sign == 0:  # Sequence length meets requirements and contains only natural amino acids

            temp.append(element)
            seq_length.append(len(temp[b]))  # Save effective sequence length
            b += 1
            element += [0] * (max_len - length)  # Pad sequence length with 0s
            data_e.append(element)
            label_e.append(label[i])

    return torch.LongTensor(np.array(data_e)), torch.LongTensor(np.array(label_e)), torch.LongTensor(
        np.array(seq_length))

def data_load(train_direction=None, test_direction=None, batch=None, subtest=True, CV=False,threshold_percentile=None):
    # Load data from target path
    dataset_train, dataset_test = [], []
    dataset_subtest = None
    weight = None
    # Load data
    train_seq_data, train_seq_label, max_len_train, min_len_train = getSequenceData(train_direction)
    test_seq_data, test_seq_label, max_len_test, min_len_test = getSequenceData(test_direction)
    print(f"max_length_train:{max_len_train}")
    print(f"min_length_train:{min_len_train}")
    print(f"max_length_test:{max_len_test}")
    print(f"min_length_test:{min_len_test}")

    x_train, y_train, train_length = PadEncode(train_seq_data, train_seq_label, max_len_train)
    x_test, y_test, test_length = PadEncode(test_seq_data, test_seq_label, max_len_test)
    # Calculate class weights
    if CV is False:  # Do not perform five-fold cross validation
        # Create datasets
        train_data = TensorDataset(x_train, train_length, y_train)
        test_data = TensorDataset(x_test, test_length, y_test)
        dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=True))
        dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=True))

        # Construct test subset
        if subtest:
            dataset_subtest = []
            for i in range(5):  # Randomly extract 80% from the test set as a subset, repeat 5 times to get 5 subsets
                sub_size = int(0.8 * len(test_data))
                _ = len(test_data) - sub_size
                subtest, _ = torch.utils.data.random_split(test_data, [sub_size, _])
                sub_test = DataLoader(subtest, batch_size=batch, shuffle=True)
                dataset_subtest.append(sub_test)
    else:
        cv = KFold(n_splits=5, random_state=42, shuffle=True)
        # Construct training and test sets for five-fold cross validation
        for split_index, (train_index, test_index) in enumerate(cv.split(x_train)):
            sequence_train, label_train, length_train = x_train[train_index], y_train[train_index], \
                                                        train_length[train_index]
            sequence_test, label_test, length_test = x_train[test_index], y_train[test_index], train_length[
                test_index]
            train_data = TensorDataset(sequence_train, length_train, label_train)
            test_data = TensorDataset(sequence_test, length_test, label_test)
            dataset_train.append(DataLoader(train_data, batch_size=batch, shuffle=True))
            dataset_test.append(DataLoader(test_data, batch_size=batch, shuffle=True))
    return dataset_train, dataset_test, dataset_subtest, weight