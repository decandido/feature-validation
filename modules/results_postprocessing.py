"""Functions for displaying and processing results"""
import numpy as np
import torch as t


def calculate_embeddings(model,
                         X_in,
                         y_in,
                         max_samples=1000,
                         batch_size=1000):
    """Function to calculate the embeddings of the different models"""
    # Calculate the output
    num_samples = X_in.shape[0]
    if num_samples > max_samples:
        layer_output = []
        for x_tmp in X_in.split(split_size=batch_size):
            layer_output.append(model.get_features(x_tmp).detach().numpy())
        # Create a numpy array out of this list
        layer_output = np.vstack(layer_output)
    else:
        layer_output = model.get_features(X_in).detach().numpy()
    # Store the embeddings per class
    embeddings_per_class = dict()
    labels = y_in.unique().numpy()
    # Calculate the mean and covariance matrices for each class and the tied
    # covariance matrix
    mean_per_class = dict()
    cov_per_class = dict()
    for label in labels:
        class_inds = (y_in == label).numpy()
        embeddings_per_class['class'+str(label)] = layer_output[class_inds, :]
        # Calculate means
        mean_per_class[f'class_{label}'] = layer_output[class_inds, :].mean(axis=0)
        cov_per_class[f'class_{label}'] = np.cov(layer_output[class_inds, :].T)
    cov_per_class['tied'] = np.cov(layer_output.T)


    return dict(embedding=layer_output), embeddings_per_class, mean_per_class, cov_per_class


def get_correct_predictions(model,
                            X_in,
                            y_in,
                            max_samples=1000):
    """Calculate which predictions were correct, and which were incorrect."""

    # Predict using the model
    num_samples = X_in.shape[0]
    if num_samples > max_samples:
        y_pred = []
        for xBatch in X_in.split(split_size=max_samples):
            y_pred.append(model(xBatch).detach())
        y_pred = t.cat(y_pred, dim=0)
    else:
        y_pred = model(X_in).detach()
    # Check which results are correct
    correct = (y_pred.argmax(1) == y_in).numpy()
    return dict(correct_inds=correct,
                prediction=y_pred.argmax(1).numpy(),
                label=y_in.numpy())


def get_name_ext(input):
    """Split the current input at the extension if possible"""
    try:
        name, ext = input.split('.')
    except ValueError as e:
        print(e)
        name, ext = -1, -1

    return name, ext

