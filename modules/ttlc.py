"""Class to define CNNs to learn the TTLC in Torch"""
import torch as t
import torch.nn as nn
from scipy import io as spio
from modules import torch_helper as th



class cnnTTLC(nn.Module):
    """Class for the Convolutional TTLC Classifiers in PyTorch"""

    # """foo = cnnLCMP(global_avg_pool=True,
    #               hidden_layers=[nn.Conv1d,
    #                              nn.Conv1d,
    #                              nn.Conv1d],
    #               embedding_dim=110,
    #               out_features=[10, 50,110],
    #               in_features=[11,10, 50],
    #               filter_dims=[5,3,2],
    #               num_channels=11,
    #               cnnType='standard')"""

    def __init__(self,
                 hidden_layers=None,
                 filter_dims=None,
                 in_features=None,
                 out_features=None,
                 hidden_args=None,
                 output_layer=None,
                 in_classifier=None,
                 out_classifier=None,
                 num_channels=None,
                 activation=None,
                 dense_activation=None,
                 output_activation=None,
                 pooling=None,
                 pooling_args=None,
                 embedding_dim=None,
                 bool_gap=True,
                 bool_batch_norm=False,
                 bool_use_bias=False,
                 dropout_rate=None,
                 cnnType=None,
                 name=None,
                 bool_normalise_output=False,
                 alpha_triplet=2.0):
        """Initialise a CNN classifier for the TTLC.

        The following arguments are required to create a feature extraction
        and classification model in Torch

        Args:
            hidden_layers: list of torch.nn layers for each of the feature
            extraction blocks, e.g., [nn.Conv1d, nn.Conv1d]
            filter_dims: list of filter dimensions for the feature extraction
            blocks, e.g., [7, 5]
            in_features: list of input sizes(#channels or # of neurons)
            for each of the feature extraction blocks
            out_features: list of output sizes (#channels or # of neurons)
            for each of the feature extraction blocks
            output_layer: list of torch.nn layers for the output classifier
            in_classifier: list of input sizes for the classifier
            out_classifier: list of input sizes for the classifier
            num_channels: number of input channels for the multi-variate time series data
            activation: string for the activation function in the feature
            extraction blocks, e.g., 'relu'
            dense_activation: string for the activation function in the
            classifier block, e.g., 'relu'
            output_activation: string for the activation at the output of the
            classifier
            pooling: torch.nn layer for pooling if requires (Default = None)
            pooling_args: dictionary of arguments for the pooling layer
            embedding_dim: int of the feature embedding dimension before
            classificatoin
            bool_gap: boolean of whether to use Global Average Pooling (GAP)
            or simply flatten the extracted features from the feature
            extraction blocks
            bool_batch_norm: boolean of whether to use batch normalisation
            bool_use_bias: boolean of whether to use a bias term in all layers
            dropout_rate: float of the dropout rate in all layers (Default =
            None, which leads to no dropout)
            cnnType: string to indicate type of feature extraction being
            used, either 'standard', 'multi_channel' or 'dense'
            name: string for the feature extraction name
        """
        super(cnnTTLC, self).__init__()
        # Intialise the network parameters
        # Define the hidden layer architecture for the feature extraction part
        if hidden_layers is None:
            self.hidden_layers = [nn.Conv1d,
                                  nn.Conv1d]
        else:
            self.hidden_layers = hidden_layers

        # Define the number of parallel channels
        if num_channels is None:
            self.num_channels = 11
        else:
            self.num_channels = num_channels

        # Get the embedding dimension
        if embedding_dim:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = 100

        # Define the type of activation to be used
        if activation is None:
            self.activation = 'relu'
        else:
            self.activation = activation

        # Define the type of activation to be used
        if dense_activation is None:
            self.dense_activation = 'relu'
        else:
            self.dense_activation = dense_activation

        # Define the type of activation to be used
        if output_activation is None:
            self.output_activation = 'softmax'
        else:
            self.output_activation = output_activation

        # Define parameters for filters
        if filter_dims is None:
            self.filter_dims = [5,
                                3]
        else:
            self.filter_dims = filter_dims

        # Set the number of input and output channels for each of the
        # convolutional layers
        if in_features is None:
            self.in_features = [self.num_channels,
                                10]
        else:
            self.in_features = in_features

        if out_features is None:
            self.out_features = [10,
                                 self.embedding_dim]
        else:
            self.out_features =out_features

        # Parameters for the classifier part of the network
        if output_layer is None:
            self.output_layer = [nn.Linear,
                                 nn.Linear]
        else:
            self.output_layer = output_layer

        # Set the number of neurons for the classifier layers
        if in_classifier is None:
            self.in_classifier = [self.embedding_dim,
                                 100]
        else:
            self.in_classifier = in_classifier
        if out_classifier is None:
            self.out_classifier = [100,
                                   17]
        else:
            self.out_classifier = out_classifier

        # Boolean whether we want to include a bias or not in all layers of
        # the model (default = False)
        self.bool_use_bias = bool_use_bias

        # Define the default pooling, dropout_rate and batch normalising
        # layer. If no value was passed, then these will be ignored when
        # building the model
        self.pooling = pooling
        if pooling_args is None:
            self.pooling_args = dict()
        else:
            self.pooling_args = pooling_args
        self.dropout_rate = dropout_rate
        # Boolean to use the batch norm before the convolutions (default: True)
        self.batch_norm = bool_batch_norm

        # Set boolean for whether to use a flattening or global average
        # poooling before the dense layers
        self.bool_gap = bool_gap

        # Set the name of the CNN
        if name is None:
            self.name = 'cnn'
        else:
            self.name = name

        # Set the type of CNN, e.g., multi_channel, standard or dense
        if cnnType is None:
            self.cnnType = 'standard'
        else:
            self.cnnType = cnnType

        # Set a boolean whether the output should be normalised or not
        self.bool_normalise_output = bool_normalise_output
        self.alpha_triplet = alpha_triplet

        # Ensure that there are the same number of filters as hidden layers
        if not self.cnnType == 'dense':
            assert len(self.filter_dims) == len(self.hidden_layers)
        assert len(self.out_features) == len(self.in_features)
        assert len(self.in_features) == len(self.hidden_layers)
        assert len(self.out_classifier) == len(self.output_layer)
        assert len(self.in_classifier) == len(self.out_classifier)

        # Create a feature list
        self.createHiddenArgs(hidden_args=hidden_args)
        # First we build the feature extraction layers
        if self.cnnType == 'multi_channel':
            featureExtractionTmp = list()
            for _ in range(self.num_channels):
                featureExtractionTmp.append(self.buildFeatureExtraction())
            # This should be a list of nnModuleLists so we will make a list
            # of lists out of it
            self.featureExtraction = nn.ModuleList(featureExtractionTmp)
        else:
            # For the normal CNNs and DNN architectures, we can simply build
            # the feature extracter from scratch
            self.featureExtraction = self.buildFeatureExtraction()

        # Create classifier part of the network
        self.classifier = self.buildClassifier()

    def forward(self, input):
        """Forward pass through feature extraction and classification parts"""
        # First we apply the feature extraction
        x = self.get_features(input)
        # After feature extraction, we want to classify the data
        for layer in self.classifier:
            x = layer(x)
        return x

    def get_features(self, x):
        """Function to extract the features"""
        if self.cnnType == 'multi_channel':
            # Need to figure out if I should put the channels at the end or where
            # we want to processes each input channel individually
            x_stacked = list()
            for channel_id, channel in enumerate(self.featureExtraction):
                x_channel = x[:, channel_id, :]
                for layer_id, layer in enumerate(channel):
                    if layer_id == 0:
                        x_channel = layer(
                            x_channel.unsqueeze(dim=1)
                        )
                    else:
                        x_channel = layer(x_channel)
                x_stacked.append(x_channel)
            # Concatenate the inputs for each channel
            x = t.cat(x_stacked, dim=1)
        else:
            for layer in self.featureExtraction:
                x = layer(x)
        return x

    def buildFeatureExtraction(self):
        """Build the feature extraction layers"""
        # there are three types of feature extraction, either (i) conv ->
        # pooling -> activation; (ii) input channel separately; (iii) dense
        # layers
        featureExtraction = list()
        # Combine the feature extraction layers together
        for layer, hidden_args in zip(self.hidden_layers,
                                      self.hidden_args):
            # First apply the layer with the arguments in hidden_args
            featureExtraction.append(
                layer(**hidden_args)
            )
            # Next apply batch normalisation, dropout and pooling if
            # required
            if self.cnnType == 'standard' or self.cnnType == 'multi_channel':
                if self.batch_norm:
                    featureExtraction.append(nn.BatchNorm1d())
                if self.dropout_rate is not None:
                    featureExtraction.append(
                        nn.Dropout(p=self.droput_rate)
                    )
                if self.pooling is not None:
                    featureExtraction.append(self.pooling(**self.pooling_args))
            # Add the activation layers
            featureExtraction.append(th.get_activation(self.activation))

        # Finally, we either want to apply global average pooling or flatten
        # the output of the channels
        if not self.cnnType == 'dense':
            if self.bool_gap:
                featureExtraction.append(
                    nn.AdaptiveMaxPool1d(1)
                )
                # In Torch we need an extra flatten after the GAP layer to
                # get rid of the singleton dimension
                featureExtraction.append(
                    nn.Flatten()
                )
            else:
                featureExtraction.append(
                    nn.Flatten()
                )

        return nn.ModuleList(featureExtraction)

    def buildClassifier(self):
        """Build the classifer layers"""
        classifierLayers = list()
        # Loop through the dense classifier layers adding dropout as required
        i = 0
        for layer, input_dim, output_dim in zip(self.output_layer,
                                                self.in_classifier,
                                                self.out_classifier):
            classifierLayers.append(
                layer(in_features=input_dim,
                      out_features=output_dim,
                      bias=self.bool_use_bias)
            )
            # check whether this is the final layer
            if not i == (len(self.output_layer) - 1):
                classifierLayers.append(th.get_activation(self.dense_activation))
            # Increment the counter
            i += 1

        return nn.ModuleList(classifierLayers)

    def createHiddenArgs(self, hidden_args=None):
        """Create a dictionary of arguments for the hidden layers of the
        feature extractor"""
        if hidden_args is None:
            hidden_args = [dict() for _ in range(len(self.in_features))]

        self.hidden_args = list()
        if self.cnnType == 'dense':
            for input_dim, output_dim, hidden_arg in zip(self.in_features,
                                                         self.out_features,
                                                         hidden_args):
                # Loop through the lists of arguments for the hidden layers
                self.hidden_args.append(dict(in_features=input_dim,
                                             out_features=output_dim,
                                             bias=self.bool_use_bias,
                                             **hidden_arg
                                             )
                                        )
        else:
            for input_dim, output_dim, kernel_dim, hidden_arg in zip(
                                                         self.in_features,
                                                         self.out_features,
                                                         self.filter_dims,
                                                         hidden_args):
                # Loop through the lists of arguments for the hidden layers
                self.hidden_args.append(dict(in_channels=input_dim,
                                             out_channels=output_dim,
                                             kernel_size=kernel_dim,
                                             bias=self.bool_use_bias,
                                             **hidden_arg
                                             )
                                        )


    def normalise(self,
                  x: t.Tensor) -> t.Tensor:
        """Function to normalise the outputs to unit norm"""
        return t.nn.functional.normalize(x)
