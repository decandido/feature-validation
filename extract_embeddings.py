"""Feature Validation extract results"""
import os
import importlib
import torch
import pickle as pkl
import scipy.io as spio
import pandas as pd
from modules import data_processing
from modules import results_postprocessing as rpp
from modules import utils
from modules import config


def bool_only_these_files(name):
    # Boolean to only select certain models for example, certain k-folds or model architectures
    # ('0' in name) or True
    return True

def get_history(history_in):
    """Return a dictionary of the history and params from the Keras history."""
    return dict(history=history_in.history,
                params=history_in.params)


def load_embeddings(embeddings_dir,
                    name,
                    kFold):
    """Load the embeddings for kFold"""
    #
    in_data = name + '_kFold' + str(kFold) + '_full.mat'
    PATH = os.path.join(embeddings_dir, in_data)
    # Load the data from the matlab file
    input_data = spio.loadmat(PATH)

    input_data = input_data['embedding']

    # Load the labels too
    in_data = name + '_kFold' + str(kFold) + '_correct_labels.mat'

    PATH = os.path.join(embeddings_dir, in_data)
    # Load the data from the matlab file
    input_labels = spio.loadmat(PATH)

    return input_data, input_labels

def save_embeddings(parargs,
                    scope: str,
                    params_file: str,
                    bool_embed_train: bool = False):

    # Booleans
    bool_save = parargs.save
    bool_embed_train = bool_embed_train  # Need this for k-means clustering

    # Whether or not we want to use the data defined in the params file
    bool_use_params_data = False

    # Load trained models and histories
    date_dir = parargs.dir
    # Get the directory names where the results are stored
    results_dir, load_dir, weight_dir, embeddings_dir, eval_dir, cluster_dir = \
        utils.get_result_directories(date_dir=date_dir,
                                     scope=scope)

    # Create a new folder for the training data embedding
    if bool_embed_train:
        embeddings_dir = os.path.join(embeddings_dir, 'training_data')
        utils.check_dir(embeddings_dir)

    # Import the parameter file used for the simulation
    params = importlib.import_module(results_dir.replace('/', '.') +
                                     '.' + date_dir +
                                     '.' + params_file)
    # Load HighD data
    PATH = params.data_path
    if bool_use_params_data:
        file_name = params.training_data
    else:
        file_name = '21_03_04_highD_150_balanced_spb2.mat'
    data = data_processing.load_training(dir_name=PATH,
                                         file_name=file_name)

    # Extract the training data
    if 'balanced' in file_name:
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

    # Extract features
    features = params.features
    X_train = X_train[:, :, features]
    X_test = X_test[:, :, features]

    # we should map the numpy arrays into torch tensors
    X_train, X_test, y_train, y_test = map(
        torch.tensor, (X_train, X_test, y_train, y_test)
    )
    # need to permute the data to comply with the torch ordering and data types
    X_train = X_train.permute(0, 2, 1).double()
    X_test = X_test.permute(0, 2, 1).double()
    y_test = y_test.long().argmax(1)
    y_train = y_train.long().argmax(1)

    # Get the names of all files in the load_dir
    files = os.listdir(load_dir)
    weight_files = os.listdir(weight_dir)
    # Automate the loading process
    models = dict()
    histories = list()
    for file in weight_files:
        # Try to split the name and file extention
        name, ext = rpp.get_name_ext(file)
        if name == -1:
            continue
        # Load the models and save them in the models list
        if (ext == 'h5' or ext == 'pt') and bool_only_these_files(name):
            _, model_name = name.split('weights_')
            print('Loading: ' + model_name)

            # Load the torch model
            PATH = os.path.join(weight_dir, file)
            model = torch.load(PATH,
                                  map_location=config.device)
            # Make sure we are in the evaluation mode
            model.eval()
            # Reshape the training data if necessary
            if bool_embed_train:
                if 'dense' in model_name:
                    X_tmp = X_train.reshape(X_train.shape[0], -1)
                else:
                    X_tmp = X_train
                # The labels are always the same
                y_tmp = y_train
            else:
                if 'dense' in model_name:
                    X_tmp = X_test.reshape(X_test.shape[0], -1)
                else:
                    X_tmp = X_test
                # Get the test labels
                y_tmp = y_test

            # Embed the data
            print('Embedding: ' + model_name)
            embedding_all_classes, tmp_emb, means, covs = \
                rpp.calculate_embeddings(model,
                                         X_tmp,
                                         y_tmp)
            # Save the embeddings per class in a dictionary
            # embeddings[model_name] = tmp_emb
            # Create name of embedding
            if bool_save:
                PATH = os.path.join(embeddings_dir, model_name + '.mat')
                print('Saving ' + model_name + ' embeddings to: ' +
                      PATH)
                spio.savemat(PATH, tmp_emb)
                PATH = os.path.join(embeddings_dir, model_name +
                                    '_full.mat')
                print('Saving ' + model_name + ' embeddings to: ' +
                      PATH)
                spio.savemat(PATH, embedding_all_classes)
                PATH = os.path.join(embeddings_dir, model_name +
                                    '_means.mat')
                print('Saving ' + model_name + ' embeddings to: ' +
                      PATH)
                spio.savemat(PATH, means)
                PATH = os.path.join(embeddings_dir, model_name +
                                    '_covs.mat')
                print('Saving ' + model_name + ' embeddings to: ' +
                      PATH)
                spio.savemat(PATH, covs)

            # Calculate the correct predicitons
            correct = rpp.get_correct_predictions(model,
                                                  X_tmp,
                                                  y_tmp)
            if bool_save:
                print(
                    'Saving ' + model_name + ' correct predictions')
                PATH = os.path.join(embeddings_dir,
                                    model_name + '_correct_labels.mat')
                spio.savemat(PATH, correct)


def cluster_embeddings(parargs,
                       scope: str,
                       params_file: str,
                       bool_embed_train: bool=False,
                       bool_correct_clusters: bool=False,
                       ):
    # Booleans
    bool_plot = parargs.plot
    bool_save = parargs.save

    # Parameters for k-means training
    num_max_clusters = 21
    num_iters_kmeans = 1
    num_iters_kmeans_sklearn = 100
    num_jobs = parargs.parallel

    # Choose a name to save the results with
    data_type = 'training_data' if bool_embed_train else 'test_data'
    save_name = 'kmeans_' + str(num_max_clusters) + '_' + data_type
    # Get the name of the date directory
    date_dir = parargs.dir

    results_dir, load_dir, weight_dir, embeddings_dir, eval_dir, _ = \
        utils.get_result_directories(date_dir=date_dir,
                                     scope=scope)

    # Import the parameter file used for the simulation
    params = importlib.import_module(results_dir.replace('/', '.') +
                                     '.' + date_dir +
                                     '.' + params_file)

    # Create a new folder for the training data embedding
    if bool_embed_train:
        embeddings_dir = os.path.join(embeddings_dir, data_type)
        utils.check_dir(embeddings_dir)

    # Get the number of kfolds
    kFolds = params.kFolds
    loop_kfolds = range(kFolds)

    # Get a list of the embedding files stored in the embeddings directory
    assert len(os.listdir(embeddings_dir)) > 0

    # Create a new folder to save the clustering results in
    cluster_dir = os.path.join(load_dir, 'clustering')
    utils.check_dir(cluster_dir)
    cluster_dir_mat = os.path.join(cluster_dir, 'mat_files_all')
    utils.check_dir(cluster_dir_mat)

    # Extract the names of the different CNN models
    model_names = [i for i in params.models.keys()]

    # Initialise a Pandas dataframe and a dictionary to store the data in
    columns = ['Model',
               'kFold',
               'rand_ind',
               'fowlkes_mallows_ind',
               'sil_score_Ypred',
               'sil_score_kMeans',
               'mutual_info',
               'num_kmeans_clusters',
               'loop_kmeans']

    df_tmp = pd.DataFrame(columns=columns)
    results_clusters = dict()
    for name in model_names:
        # Only do the clustering on certain embeddings
        if not bool_only_these_files(name):
            continue
        for i in loop_kfolds:
            # Print the current model name
            print('Clustering: ' + name + ' ' + str(i))

            # Load the embeddings for the k-means clustering
            X, y = load_embeddings(embeddings_dir=embeddings_dir,
                                   name=name,
                                   kFold=i)

            # Run the k-means clustering
            kMeans_results, kMeans_mat, kMeans_centers_mat, df_tmp = \
                utils.perform_kMeans(X_input=X,
                                     y_pred=y['prediction'],
                                     y_labels=y['label'],
                                     df_tmp=df_tmp,
                                     num_iters_kmeans=num_iters_kmeans,
                                     num_max_clusters=num_max_clusters,
                                     num_iters_kmeans_sklearn=num_iters_kmeans_sklearn,
                                     num_jobs=num_jobs,
                                     model_name=name,
                                     kFold=i,
                                     bool_correct_clusters=bool_correct_clusters,
                                     )

            # store the results for the clustering
            results_clusters[name + '_kFold' + str(i)] = kMeans_results
            # Save the contigency matrices, labels and kMeans labels
            if bool_save:
                PATH = os.path.join(cluster_dir_mat,
                                    save_name + name + '_kFold'
                                    + str(i) + '.mat')

                spio.savemat(PATH,
                             kMeans_mat)
                # Store the k-Means centers too
                PATH = os.path.join(cluster_dir_mat,
                                    save_name + name + '_kFold'
                                    + str(i) + '_kMean_centers.mat')

                spio.savemat(PATH,
                             kMeans_centers_mat)


            # Save the dataframe with the different indeces
            if bool_save:
                PATH = os.path.join(cluster_dir,
                                    save_name + '.csv')
                # Save the dataframe to the folder
                df_tmp.to_csv(PATH)


    if bool_save:
        # Store the clustering results
        PATH = os.path.join(cluster_dir, save_name + '_kmeans.pkl')
        with open(PATH, 'wb') as file:
            pkl.dump(results_clusters, file)


if __name__ == '__main__':
    # Define parameters for simulation
    parser = utils.get_parser()
    parargs = parser.parse_args()
    bool_correct_clusters = False
    bool_embed_train = False
    # Where to save the results
    scope = 'ttlc-highd'
    # Where is the network param file stored?
    params_file = 'params'

    # Extract the feature embeddings and save them
    save_embeddings(parargs,
                    scope=scope,
                    params_file=params_file,
                    bool_embed_train=bool_embed_train)

    # Cluster the feature embeddings
    cluster_embeddings(parargs,
                       scope=scope,
                       params_file=params_file,
                       bool_embed_train=bool_embed_train,
                       bool_correct_clusters=bool_correct_clusters)