"""Function to read in embedding files and calculate the UMAP"""
import os
import gc
import importlib
import umap
import warnings
import scipy.io as spio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules import results_postprocessing as rpp
from modules import utils
# Ignore warning from NumbaPerformance
warnings.filterwarnings('ignore')


def plot_umap(data,
              labels,
              fit=None,
              n_neighbors=15,
              min_dist=0.1,
              n_components=2,
              metric='euclidean',
              title='',
              save_dir=None,
              bool_save_csv=False):
    """Use UMAP to fit the data and plot the data

    Adapted code from the UMAP documentation/tutorial:
    https://umap-learn.readthedocs.io/en/latest/parameters.html
    """
    # First create a saveable title
    title_save = title.replace('.', ' ').replace(' ', '_')
    # Initialize pandas columns
    columns = ['u1', 'u2', 'labels']
    if fit is None:
        fit = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric
        )
        # Fit the UMAP to the input data
        u = fit.fit_transform(data)
        # Pandas
        df = pd.DataFrame(data=np.hstack((u, labels)),
                          columns=columns)
        if save_dir is not None and bool_save_csv:
            PATH = os.path.join(save_dir, title_save + '.csv')
            df.to_csv(PATH)
    else:
        # If we passed the UMAP already then we can simply pass
        # u_prime = fit.transform(data)

        print('Do nothing')
        u, centers = data
        # Transform the kMeans centers
        u_centers = fit.transform(centers)
        # Store the data in a pandas dataframe for saving
        df = pd.DataFrame(data=np.hstack((u, labels.T)),
                          columns=columns)
        # Create a dataframe for the cluster centers too
        labels_centers = np.array(['c' + str(i) for i in range(1,
                                                               len(np.unique(labels)) + 1)])
        df_centers = pd.DataFrame(data=np.hstack((u_centers, labels_centers[
                                                             :, np.newaxis])),
                                  columns=columns)
        if save_dir is not None and bool_save_csv:
            PATH = os.path.join(save_dir, title_save + '.csv')
            df.to_csv(PATH)
            PATH = os.path.join(save_dir, title_save + 'clusters.csv')
            df_centers.to_csv(PATH)



    # Use the UMAP plotting functions
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=labels)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(np.reshape(u[:, 0], -1),
                   np.reshape(u[:, 1], -1),
                   s=1,
                   c=np.reshape(labels, -1))
        # if we have centers then plot them too
        if 'centers' in locals():
            ax.scatter(np.reshape(u_centers[:, 0], -1),
                       np.reshape(u_centers[:, 1], -1),
                       s=50,
                       c='k',
                       marker='*')
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=labels, s=100)
    plt.title(title, fontsize=18)
    # Only display the plot if we don't want to save it
    if save_dir is None:
        plt.show()
    else:
        PATH = os.path.join(save_dir, title_save + '.png')
        fig.savefig(PATH, format='png')


    # Return the learned UMAP and the embeddings
    return fit, u


def bool_only_these_files(name):
    # Boolean to only select certain models
    # ('0' in name)
    return (True)

def bool_only_these_clusters(name):
    return ('7' in name) and ('17' not in name)

def plot_umaps():

    # Define parameters for simulation
    parser = utils.get_parser()
    parargs = parser.parse_args()
    # Booleans
    bool_save = parargs.save
    bool_clusters = True
    bool_all_data = True
    bool_save_csv = True
    # Umap labels
    umap_label = 'prediction' # or use 'label' to see which label was
    # predicted
    load_name = 'kmeans_21_test_data' # Should be the same as save_name in `extract_embeddings.py`

    # Some parameters to test the UMAPs with
    metrics = ['euclidean']
    min_dists = [0.5]
    num_neigbours = [15]

    # Load trained models and histories
    # Get the name of the date directory
    date_dir = parargs.dir
    scope = 'ttlc-highd'
    params_file = 'params'

    # Get the directory names where the results are stored
    results_dir, load_dir, weight_dir, embeddings_dir, eval_dir, cluster_dir = \
        utils.get_result_directories(date_dir=date_dir,
                                     scope=scope)

    # Initialise a folder to store the UMAPs
    umaps_dir = os.path.join(load_dir, 'umap')
    utils.check_dir(umaps_dir)

    # Get the names of all files in the load_dir
    files = os.listdir(weight_dir)
    # Automate the loading process
    for file in files:
        # load the MAT files of the embeddings, and the labels
        # Try to split the name and file extention
        name, ext = rpp.get_name_ext(file)
        if name == -1:
            continue
        if (ext == 'h5' or ext == 'pt') and bool_only_these_files(name):
            _, model_name = name.split('weights_')
            # Display which method is currently being used
            print('Umapping ' + model_name)

            # Create a path for the current UMAP
            super_model, _ = model_name.split('_')
            if bool_save:
                umaps_dir_tmp = os.path.join(umaps_dir, super_model)
                utils.check_dir(umaps_dir_tmp)
            else:
                umaps_dir_tmp = None

            # Load the models and save them in the models list
            PATH = os.path.join(embeddings_dir,
                                model_name + '_full.mat')
            latent_embeddings = spio.loadmat(PATH)
            # Load the labels
            PATH = os.path.join(embeddings_dir, model_name +
                                '_correct_labels.mat')
            labels = spio.loadmat(PATH)
            # Load the clustering results
            PATH = os.path.join(cluster_dir, 'mat_files_all',
                                load_name + model_name +
                                '.mat')
            clusters = spio.loadmat(PATH)
            # Load the clustering centers
            PATH = os.path.join(cluster_dir, 'mat_files_all',
                                load_name + model_name + '_kMean_centers.mat')
            cluster_centers = spio.loadmat(PATH)

            # Run UMAP on the embeddings for different possible parameters
            for n in num_neigbours:
                for d in min_dists:
                    for metric in metrics:
                        # Create a title for the current plot
                        umap_title = model_name + ' ' + metric +\
                                     ' n_neighbors {} min_dist {}'.format(n, d)

                        fit, umap_embedded = plot_umap(data=latent_embeddings['embedding'],
                                                      labels=labels[umap_label].T,
                                                      n_neighbors=n,
                                                      min_dist=d,
                                                      metric=metric,
                                                      title=umap_title,
                                                      save_dir=umaps_dir_tmp,
                                                      bool_save_csv=bool_save_csv)

                        # Extract the clustered data, i.e., plot the UMAPs with the k-means
                        # clustering labels
                        if bool_clusters:
                            if bool_all_data:
                                inds = np.ones(labels['correct_inds'].shape[1]).astype(bool)
                            else:
                                inds = np.reshape(labels['correct_inds'].T.astype(bool),
                                              -1)
                            cluster_data = umap_embedded[inds, :]
                            for cluster_name, cluster in clusters.items():
                                # Extract the cluster labels and the correct
                                # data for this embedding
                                if '__' in cluster_name or 'matrix' in \
                                        cluster_name or 'y_pred' in cluster_name:
                                    continue
                                if not bool_only_these_clusters(cluster_name):
                                    continue
                                # Create a new title for the UMAP
                                umap_title_tmp = umap_title + ' ' + cluster_name
                                cluster_center = cluster_centers[cluster_name]
                                _, _ = plot_umap(data=(cluster_data,
                                                       cluster_center),
                                                labels=cluster,
                                                fit=fit,
                                                title=umap_title_tmp,
                                                save_dir=umaps_dir_tmp,
                                                bool_save_csv=bool_save_csv)

                        #clear out the garbage
                        gc.collect()


if __name__ == '__main__':
    # Plot and save the UMAP emeddings
    plot_umaps()