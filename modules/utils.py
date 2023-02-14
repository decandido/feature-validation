"""Various utility functions."""
import os
import datetime
import argparse
import time
from tqdm import tqdm
import torch as t
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from functools import wraps

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def create_save_folder(dir_base=None):
    """Function to get the current time and create a folder to save results"""
    if dir_base is None:
        # Default save folder is './results/
        dir_base = os.path.sep.join(['.', 'results'])
    # Get the current time and date
    now = datetime.datetime.now()
    time_now = '{}-{}-{}-{}_{}_{}'.format(now.year,
                                          now.month,
                                          now.day,
                                          now.hour,
                                          now.minute,
                                          now.second)
    dir_name = os.path.sep.join([dir_base, time_now])

    # Create a new directory
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    return dir_name


def get_parser():
    """Function to define the parser arguments needed for training"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--save',
        help='boolean for saving data',
        type=bool,
        default=False
    )
    parser.add_argument(
        '-l',
        '--load',
        help='boolean for loading data',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--plot',
        help='boolean for plotting',
        type=bool,
        default=False
    )
    parser.add_argument(
        '-t',
        '--train',
        help='boolean for training',
        type=bool,
        default=False
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help='boolean for verbose training',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--perm_labels',
        help='boolean for permuting labels',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--gpu',
        help='which gpu should be used',
        type=str,
        default=None
    )
    parser.add_argument(
        '-p',
        '--parallel',
        help='number of parallel processes',
        type=int,
        default=1
    )
    parser.add_argument(
        '-d',
        '--dir',
        help='directory name for loading results',
        type=str,
        default='./'
    )
    parser.add_argument(
        '--scope',
        help='scope of results',
        type=str,
        default='Torch'
    )
    parser.add_argument(
        '--save_name',
        help='filename',
        type=str,
        default='save'
    )
    return parser


def setup_gpus_pytorch(device_id='0'):
    """Function to initialise pytorch GPU device"""
    if t.cuda.is_available():
        device = t.device('cuda:{}'.format(device_id))
    else:
        device = t.device('cpu')

    return device


def sec_to_hours(seconds: float):
    """"Convert number of seconds to a string hh:mm:ss."""
    # hours
    h = seconds // 3600
    # remaining seconds
    r = seconds % 3600
    return '{:.0f}:{:02.0f}:{:02.0f}'.format(h, r // 60, r % 60)


def get_result_directories(date_dir=None,
                           scope=None):
    """Get the directories for the results"""
    # Check whether we are running this on the server or local
    volume_dir = os.path.join('TTLC')
    # Make sure we can run this on the servers
    if not os.path.exists(volume_dir):
        # alternative if we want to use the data locally
        volume_dir = '.'
    # Set the scope of experiments
    if scope is None:
        scope = 'ttlc-highd'
    results_dir = os.path.join('results', scope)
    load_dir = os.path.join(volume_dir, results_dir, date_dir)
    weight_dir = os.path.join(load_dir, 'weights')
    embeddings_dir = os.path.join(load_dir, 'embeddings')
    eval_dir = os.path.join(load_dir, 'eval')
    cluster_dir = os.path.join(load_dir, 'clustering')

    return results_dir, load_dir, weight_dir, embeddings_dir, eval_dir, \
           cluster_dir

def check_dir(dir):
    """Check whether directory exists, and create it if needed."""
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def perform_kMeans(X_input,
                   y_pred,
                   y_labels,
                   df_tmp,
                   num_iters_kmeans,
                   num_iters_kmeans_sklearn,
                   num_jobs,
                   num_max_clusters,
                   model_name,
                   bool_correct_clusters=False,
                   inds_correct=None,
                   kFold=None,
                   epoch=None):
    """Calculate the kMeans clustering on the embeddings"""

    kMeans_results = dict()
    kMeans_mat = dict()
    kMeans_centers_mat = dict()
    for tmp_clusters in tqdm(range(2, num_max_clusters + 1)):
        for kmeans_iter in range(num_iters_kmeans):
            kmeans_tmp = KMeans(n_clusters=tmp_clusters,
                                n_init=num_iters_kmeans_sklearn,
                                n_jobs=num_jobs).fit(X_input)
            kMeans_results['Clusters_' + str(tmp_clusters)] = kmeans_tmp
            kMeans_mat['Clusters_' + str(tmp_clusters)] = \
                kmeans_tmp.labels_

            # Save the kMeans centers too
            kMeans_centers_mat['Clusters_' + str(tmp_clusters)] = \
                kmeans_tmp.cluster_centers_

            # sil_samples = metrics.silhouette_samples(X=X_input,
            #                                          labels=kmeans_tmp.cluster_centers_)

            # Calculate the adjusted Rand Index
            rand_pred = metrics.adjusted_rand_score(
                labels_pred=kmeans_tmp.labels_,
                labels_true=np.squeeze(y_labels)
            )
            # Calculate the Fowlkes Mallows Score
            fm_score = metrics.fowlkes_mallows_score(
                labels_pred=kmeans_tmp.labels_,
                labels_true=np.squeeze(y_labels)
            )
            # Calculate the Silhouette Coeffiecent score
            sil_score_pred = metrics.silhouette_score(X=X_input,
                                                      labels=np.squeeze(y_labels))
            sil_score_kmeans = metrics.silhouette_score(X=X_input,
                                                        labels=kmeans_tmp.labels_)

            # Calcualte the normalise mutual information
            mutual_info = metrics.normalized_mutual_info_score(
                labels_true=np.squeeze(y_labels),
                labels_pred=kmeans_tmp.labels_
            )

            # Calculate the contingency matrix
            cont_tmp = metrics.cluster.contingency_matrix(
                labels_true=y_labels,
                labels_pred=kmeans_tmp.labels_
            )

            ##
            kMeans_results['Contingency_matrix_clusters_' \
                           + str(tmp_clusters)] = cont_tmp
            kMeans_mat['Contingency_matrix_clusters_' \
                       + str(tmp_clusters)] = cont_tmp
            # Store the current results into the temporary pandas
            # dataframe

            dict_tmp = {'Model': model_name,
                        'kFold': str(kFold),
                        'rand_ind': rand_pred,
                        'fowlkes_mallows_ind': fm_score,
                        'sil_score_Ypred': sil_score_pred,
                        'sil_score_kMeans': sil_score_kmeans,
                        'mutual_info' : mutual_info,
                        'num_kmeans_clusters': str(
                            tmp_clusters),
                        'loop_kmeans': str(kmeans_iter)}
            if epoch is not None:
                dict_tmp['epoch'] = epoch
            # Insert the number of data points
            if bool_correct_clusters:
                dict_tmp['datapoints'] = sum(inds_correct)
            # Add this dictionary to the dataframe
            df_tmp = df_tmp.append(dict_tmp,
                                   ignore_index=True)
    # Also store the predicted labels of the model
    kMeans_results['y_pred'] = y_pred.T
    kMeans_mat['y_pred'] = y_pred.T

    return kMeans_results, kMeans_mat, kMeans_centers_mat, df_tmp

