"""Load the training data for learning"""
import os
import numpy as np
import scipy.io as spio
import sklearn.preprocessing as skpre
from scipy import interpolate


def load_training(dir_name,
                  file_name):
    # initalise a dictionary
    data = dict()

    # load the MATLAB data using the scipy importer
    data_tmp = spio.loadmat(os.path.join(dir_name, file_name),
                            matlab_compatible=True)

    data_tmp_keys = data_tmp.keys()

    if 'labels' in data_tmp_keys:
        if data_tmp['labels'].shape[-1] > data_tmp['logits'].shape[0]:
            data_tmp['logits'] = data_tmp['logits'].T
            data_tmp['labels'] = data_tmp['labels'].T

        data['data'] = data_tmp['logits']
        data['labels'] = data_tmp['labels']
    elif 'X_train' in data_tmp_keys:
        if data_tmp['X_train'].shape[-1] > data_tmp['X_train'].shape[0]:
            data_tmp['X_train'] = data_tmp['X_train'].T
            data_tmp['X_test'] = data_tmp['X_test'].T
            data_tmp['y_train'] = data_tmp['y_train'].T
            data_tmp['y_test'] = data_tmp['y_test'].T

        # Extract the training samples
        data['X_train'] = data_tmp['X_train']
        data['X_test'] = data_tmp['X_test']
        data['y_train'] = data_tmp['y_train']
        data['y_test'] = data_tmp['y_test']

    return data


def normalise_data(data,
                   bool_locations=False,
                   bool_scale_minmax=True,
                   bool_scale_std=False,
                   args_minmax=None,
                   args_std=None,
                   scaler=None):
    """Normalise the training data as required."""
    if args_minmax is None:
        args_minmax = dict()
    if args_std is None:
        args_std = dict()

    if bool_locations:
        data_ = data['data_stacked']
    else:
        data_ = data['data']

    # get the number of features
    num_features = data_.shape[1]
    num_frames = data_.shape[-1]

    # Normalise the all of the data jointly
    data_stacked = np.moveaxis(data_, -1, 1).reshape((-1, num_features))

    # If we pass a scaler, we can directly scale the data
    if scaler is not None:
        data_scaled = scaler.transform(data_stacked).reshape((-1, num_frames, num_features))
    elif bool_scale_minmax:
        scaler = skpre.MinMaxScaler()
        data_scaled = scaler.fit_transform(data_stacked).reshape((-1,
                                                                  num_frames,
                                                                  num_features))
    elif bool_scale_std:
        scaler = skpre.StandardScaler()
        data_scaled = scaler.fit_transform(data_stacked).reshape((-1, num_frames, num_features))
    else:
        data_scaled = data_stacked.reshape((-1, num_frames, num_features))

    if bool_locations:
        start = 0
        for i in range(6):
            inds = range(start, start + len(data['labels' + str(i)]))
            data['data' + str(i) + '_scaled'] = np.moveaxis(data_scaled[inds, :,
                                                            :], -1, 1)
            start += len(data['labels' + str(i)])
    else:
        data['data_scaled'] = data_scaled

    return data, scaler


def get_classification_data(data,
                            num_frames,
                            frame_rate,
                            seconds_per_bin,
                            bool_augmented_labels=False):
    """Function to create classification data for training."""
    # Calculate how many classes we can have
    num_classes = (num_frames // frame_rate) // seconds_per_bin

    # Extract the different
    X_ = None
    Y_ = None
    for ttlc in range(num_classes):
        extract_frames = range(num_frames -
                               (ttlc + 1) * frame_rate * seconds_per_bin,
                               num_frames - ttlc * seconds_per_bin * frame_rate)
        tmp_X = data['data_scaled'][:, extract_frames, :]
        # Get the labels and multiply them with the time to lane change so
        # a lane change left with 3s ttlc will have the label +3 and a right
        # lane change with ttlc = 2s with have the label -2.  A lane keep
        # will always have the label 0.
        tmp_Y = data['labels'] * (ttlc + 1)
        if X_ is None and Y_ is None:
            X_ = tmp_X
            Y_ = tmp_Y
        else:
            X_ = np.vstack((X_, tmp_X))
            Y_ = np.vstack((Y_, tmp_Y))

    # Onehot encode the labels
    enc = skpre.OneHotEncoder(categories='auto')
    Y_onehot = enc.fit_transform(Y_)
    Y_onehot = Y_onehot.toarray()

    X_balanced, Y_balanced = get_equal_class_sizes(X_,
                                                   Y_onehot)

    # We will agument the labels of the balanced dataset
    if bool_augmented_labels:
        Y_balanced = get_augmented_labels(Y_balanced)

    return (X_, Y_, Y_onehot, enc), (X_balanced, Y_balanced)


def get_equal_class_sizes(X_in,
                          Y_in,
                          bool_random=False):
    """Function to evenly split the Training data for each class"""
    # First, we need to calculate how many samples we have per class
    num_per_class = np.sum(Y_in, axis=0).T

    # Now, we select the minimum number to balance the classes
    min_num_per_class = np.min(num_per_class)
    del_per_class = num_per_class - min_num_per_class
    # Now we select exactly min_num_per_class samples from each class to
    # create our balances dataset
    for i in range(Y_in.shape[1]):
        indices,  = np.nonzero(Y_in[:, i] == 1)
        to_del = int(del_per_class[i])
        # We can skip the loops for the class with the least number of samples
        if to_del == 0:
            continue
        if bool_random:
            np.random.shuffle(indices)
        else:
            del_indices = indices[:to_del]

        X_in = np.delete(X_in, del_indices, axis=0)
        Y_in = np.delete(Y_in, del_indices, axis=0)


    return X_in, Y_in


def get_augmented_labels(Y_in,
                         weight=-1):
    """Function to augment the data labels to penalize false alarm rate"""
    num_labels = Y_in.shape[-1]
    keep_ind = int(np.floor(num_labels/2))
    # Go through the different labels
    for i in range(num_labels):
        # Get the indicies of
        inds = Y_in[:, i] == 1
        if i < keep_ind:
            Y_in[inds, keep_ind+1: ] = -1
        elif i > keep_ind:
            Y_in[inds, :keep_ind] = -1

    return Y_in


def save_training_test(data,
                       dir_name,
                       file_name):
    """Function to save the training test split so this is only calculated
    one."""
    X_train, X_test, y_train, y_test = data
    data_out = dict(X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test)
    PATH = os.path.sep.join([dir_name, file_name])
    spio.savemat(PATH, data_out)


def interpolate_data(data:np.array,
                     num_frames:int,
                     num_frames_interp:int,
                     num_features:int,) -> np.array:
    """ Function to interpolate the input data so we can compare classifiers trained on the same
    length input data.
    Function adapted from Xinyang Li Forschungspraxis Code"""

    t = np.linspace(0, num_frames_interp, num_frames, endpoint=False)
    tnew = np.linspace(0, num_frames_interp, num_frames_interp, endpoint=False)
    data_out = np.zeros((data.shape[0], num_features, num_frames_interp))

    for i in range(data.shape[0]):
        for j in range(num_features):
            tem = data[i, j, :]
            f = interpolate.interp1d(t, tem)
            fnew = f(tnew)
            data_out[i, j, :] = fnew

    return data_out