"""File to save simulation parameters."""
import os
import torch as t
import torch.nn as nn
from modules import torch_helper as th
# This file will be copied to the simulation results folder so the simulation
# can be reproducible
# Booleans for whether to train, load and save the model
bool_save = True
bool_train = True
bool_load = False
bool_plot = False
bool_load_optimiser = False
kFolds = 2
random_state = 42

# Booleans for loading and scaling training data
bool_normalize = False # Do we want to renormalise the data?
bool_scale_minmax = False
bool_scale_std = False
bool_save_training_data = False
bool_interp=False
# Choose which data to use
data_path = './data/highD/'
training_data = '21_03_04_highD_150_balanced_spb2.mat'
# Results scope
scope = 'ttlc-highd'

# Some parameters for the training
epochs = 2
batch_size = 200
batch_size_test = 50
batch_size_val = 500

# Parameters for the learning optimizer
lr = 0.00005
mom = 0.5

# Percentage of data to be used for training, 1 - per_train will be used for
# validation
per_train = 0.8
per_val = 0.1
per_test = 1 - per_train

# Frame rate and total number of frames in training data
frame_rate = 25 # Use this for original HighD data
num_frames = 150
seconds_per_bin = 2
save_training_data = training_data.split('.mat')[0] + \
                     '_balanced_spb{''}.'.format(seconds_per_bin)

# Choose which features we want to train with
# e.g. if we want to learn with all features set features = range(11)
# if we wand to learn from just the distances then use
# features = range(3, 9)
features = range(10)

# Set the number of epochs which should pass before the optimiser state is
# stored
epochs_per_save = 5
save_best_only = True
early_stopping_patience = 20
early_stopping = True
# Need to specify which metric should be watched for early stopping
early_stopping_metric = 'val_accuracy'
check_point = False
# if we want to cluster the features during training to see the affect of
# clustering
cluster_per_epoch = False
feature_cluster_epochs = 10

################################################################################
## Parameters for the different CNN architectures                             ##
################################################################################
# Global parameters for all architectures
num_classes = 7
output_args = dict()
activation = 'tanh'
embedding_dim = 100
# Set the booleans for all methods, this includes whether to use the batch
# norm, dropout, a bias term per filter and the Global Average Pooling (GAP)
bool_batch_norm = False
dropout_rate = None
bool_use_bias = True
bool_gap = True
# If we don't want max pooling we can set the pooling layer to a linear
# activation layer which simply passes the signal on
pooling = None
pooling_args = None
# pooling = k.layers.MaxPool1D
output_layer = [nn.Linear]
in_classifier = [embedding_dim]
out_classifier = [num_classes]
# Activation functions of the classifier part
dense_activation = activation
output_activation = 'softmax'
filter_dims = [8,
               5,
               3]
# Calculate the output sizes for the local 1D convolutions
output_size = [seconds_per_bin * frame_rate - filter_dims[0] + 1]
for i, dim in enumerate(filter_dims[1:]):
    output_size.append(output_size[i] - dim + 1)
optimiser = t.optim.Adam
verbose = 1
# How do we want to initialise the weights?
initialiser = t.nn.init.xavier_uniform_
initialiser_args = dict(gain=t.nn.init.calculate_gain(activation))

################################################################################
## Loss Function                                                              ##
################################################################################
# loss = nn.CrossEntropyLoss
loss = nn.NLLLoss
################################################################################
## Normal 1-D CNN                                                             ##
################################################################################
normCNN_args = dict(hidden_layers=[nn.Conv1d,
                                   nn.Conv1d,
                                   nn.Conv1d],
                    num_channels=len(features),
                    activation=activation,
                    in_features=[10,
                                 211,
                                 260],
                    out_features=[211,
                                  260,
                                  embedding_dim],
                    filter_dims=filter_dims,
                    bool_gap=bool_gap,
                    output_layer=output_layer,
                    in_classifier=in_classifier,
                    out_classifier=out_classifier,
                    pooling=pooling,
                    bool_batch_norm=bool_batch_norm,
                    bool_use_bias=bool_use_bias,
                    dropout_rate=dropout_rate,
                    embedding_dim=embedding_dim)

################################################################################
## MultiChannel 1-D CNN                                                       ##
################################################################################
multiCNN_args = dict(num_channels=len(features),
                     hidden_layers = [nn.Conv1d,
                                      nn.Conv1d,
                                      nn.Conv1d],
                     activation=activation,
                     in_features=[ 1,
                                   65,
                                   102],
                     out_features=[ 65,
                                    102,
                                    int(embedding_dim//len(features))],
                     filter_dims=filter_dims,
                     bool_gap=bool_gap,
                     output_layer=output_layer,
                     in_classifier=in_classifier,
                     out_classifier=out_classifier,
                     pooling=pooling,
                     bool_batch_norm=bool_batch_norm,
                     bool_use_bias=bool_use_bias,
                     dropout_rate=dropout_rate,
                     embedding_dim=embedding_dim)

################################################################################
## DenseNN                                                                    ##
################################################################################
denseNN_args = dict(hidden_layers=[nn.Linear,
                                   nn.Linear,
                                   nn.Linear],
                    num_channels=len(features),
                    activation=activation,
                    in_features=[len(features) * frame_rate * seconds_per_bin,
                                 375,
                                 380],
                    out_features=[375,
                                  380,
                                  embedding_dim],
                    filter_dims=filter_dims,
                    bool_gap=False,
                    output_layer=output_layer,
                    in_classifier=in_classifier,
                    out_classifier=out_classifier,
                    pooling=None,
                    bool_batch_norm=bool_batch_norm,
                    bool_use_bias=bool_use_bias,
                    dropout_rate=dropout_rate,
                    embedding_dim=embedding_dim)

################################################################################
## Separable 1-D CNN                                                          ##
################################################################################
sepCNN_args = dict(hidden_layers=[th.Conv1dSeparable,
                                  th.Conv1dSeparable,
                                  th.Conv1dSeparable],
                    num_channels=len(features),
                    activation=activation,
                    in_features=[10,
                                 550,
                                 552],
                    out_features=[550,
                                  552,
                                  embedding_dim],
                    filter_dims=filter_dims,
                    bool_gap=bool_gap,
                    output_layer=output_layer,
                    in_classifier=in_classifier,
                    out_classifier=out_classifier,
                    pooling=pooling,
                    bool_batch_norm=bool_batch_norm,
                    bool_use_bias=bool_use_bias,
                    dropout_rate=dropout_rate,
                    embedding_dim=embedding_dim)


################################################################################
## MultiChannel Separable 1-D CNN                                             ##
################################################################################
multiSepCNN_args = dict( num_channels=len(features),
                         hidden_layers = [th.Conv1dSeparable,
                                          th.Conv1dSeparable,
                                          th.Conv1dSeparable],
                         activation=activation,
                         in_features=[1,
                                       128,
                                       256],
                         out_features=[128,
                                        256,
                                        int(embedding_dim//len(features))],
                         filter_dims=filter_dims,
                         bool_gap=bool_gap,
                         output_layer=output_layer,
                         in_classifier=in_classifier,
                         out_classifier=out_classifier,
                         pooling=pooling,
                         bool_batch_norm=bool_batch_norm,
                         bool_use_bias=bool_use_bias,
                         dropout_rate=dropout_rate,
                         embedding_dim=embedding_dim)

################################################################################
## Local 1-D CNN                                                              ##
################################################################################
localCNN_args = dict(hidden_layers=[th.Conv1dLocal,
                                    th.Conv1dLocal,
                                    th.Conv1dLocal],
                    num_channels=len(features),
                    activation=activation,
                    in_features=[10,
                                 20,
                                 20],
                    out_features=[20,
                                  20,
                                  embedding_dim],
                    hidden_args=[dict(output_size=sz) for sz in output_size],
                    filter_dims=filter_dims,
                    bool_gap=bool_gap,
                    output_layer=output_layer,
                    in_classifier=in_classifier,
                    out_classifier=out_classifier,
                    pooling=pooling,
                    bool_batch_norm=bool_batch_norm,
                    bool_use_bias=bool_use_bias,
                    dropout_rate=dropout_rate,
                    embedding_dim=embedding_dim)

################################################################################
## MultiChannel Local 1-D CNN                                                 ##
################################################################################
multiLocalCNN_args = dict(num_channels=len(features),
                          hidden_layers = [th.Conv1dLocal,
                                           th.Conv1dLocal,
                                           th.Conv1dLocal],
                          activation=activation,
                          in_features=[1,
                                       10,
                                       11],
                          out_features=[10,
                                        11,
                                        int(embedding_dim//len(features))],
                          hidden_args=[dict(output_size=sz) for sz in
                                       output_size],
                          filter_dims=filter_dims,
                          bool_gap=bool_gap,
                          output_layer=output_layer,
                          in_classifier=in_classifier,
                          out_classifier=out_classifier,
                          pooling=pooling,
                          bool_batch_norm=bool_batch_norm,
                          bool_use_bias=bool_use_bias,
                          dropout_rate=dropout_rate,
                          embedding_dim=embedding_dim)

################################################################################
## Dictionary of parameters                                                   ##
################################################################################
# The second argument indecates how the model should be built, i.e.,
# standard, multi_channel or dense
models = dict(
              normalCNN=[normCNN_args, 'standard'],
              separableCNN=[sepCNN_args, 'standard'],
              localCNN=[localCNN_args, 'standard'],
              multiChannelCNN=[multiCNN_args, 'multi_channel'],
              multiChannelSepCNN=[multiSepCNN_args, 'multi_channel'],
              multiChannelLocalCNN=[multiLocalCNN_args, 'multi_channel'],
              denseNN=[denseNN_args, 'dense'],
)

for name, item in models.items():
    assert item[1] == 'standard' or \
           item[1] == 'multi_channel' or \
           item[1] == 'dense', 'The model must be built has a standard CNN, ' \
                               'a multi-channel CNN, or a dense NN'
