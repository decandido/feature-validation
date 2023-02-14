"""Script to train Time-to-Lane-Change (TTLC) Predictors"""
import os
import torch
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from modules import data_processing
from modules import ttlc
from modules import torch_helper as th
from modules import utils
from modules import config
from parameters import params


def plot_history(history,
                 filename):

    x_axis = np.arange(1, len(history['val_loss_per_epoch'])+1)
    plt.plot(x_axis, history['train_loss_per_batch'][:, -1])
    plt.plot(x_axis, history['val_loss_per_epoch'])
    plt.plot(x_axis, history['train_loss_per_batch'].mean(axis=1))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(('loss', 'val_loss', 'loss_mean'))
    plt.savefig(filename)
    plt.close()


def get_history(history_in, name):
    """Return a dictionary of the history and params from the Keras history."""
    return dict(history=history_in.history,
                params=history_in.params,
                name=name)


def main():

    # Define parameters for simulation
    parser = utils.get_parser()
    parargs = parser.parse_args()

    if params.bool_save:
        # create a folder to save the data
        dir_name = utils.create_save_folder(
            dir_base='./results/' + params.scope)
        # Copy the parameter file into the new folder for future reference
        # Don't forget to change this if we train a new model, e.g., transformer
        outfile = os.path.join(dir_name, 'params.py')
        os.system('cp ./parameters/params.py {}'.format(
            outfile))

    # If we want to use the GPUs
    if parargs.gpu:
        config.device = utils.setup_gpus_pytorch(device_id=parargs.gpu)
        print('Using device = [{}]'.format(config.device))

    # Booleans to know whether to load or execute new test
    bool_save = params.bool_save
    bool_train = params.bool_train

    # Booleans for how we want to normalise the training data
    bool_normalize = params.bool_normalize
    bool_save_training_data = params.bool_save_training_data
    bool_interp = params.bool_interp

    # HighD was filmed at a frame rate of 25 Hz, so we can
    # select the last second before the lane change
    frame_rate = params.frame_rate
    seconds_per_bin = params.seconds_per_bin

    # Train/Val/Test split
    per_test = params.per_test
    per_val = params.per_val

    # Choose which features should be used e.g. if we want to learn with all features set
    # features = range(11), if we wand to learn from just the distances then use features =
    # range(3, 9)
    features = params.features

    # Load HighD data
    PATH = params.data_path
    file_name = params.training_data
    data = data_processing.load_training(dir_name=PATH,
                                         file_name=file_name)

    # Create save directories if required
    if bool_save:
        file_save_name = 'weights'
        # Save the models and evaluate them
        path_weights = os.path.sep.join([dir_name, 'weights'])
        path_eval = os.path.sep.join([dir_name, 'eval'])
        path_embeddings = os.path.sep.join([dir_name, 'embeddings'])
        path_history = os.path.sep.join([dir_name, 'history'])
        utils.check_dir(path_weights)
        utils.check_dir(path_eval)
        utils.check_dir(path_embeddings)
        utils.check_dir(path_history)

    if bool_normalize:

        # Get the number of frames from the ttlc.py
        num_frames = data['data'].shape[-1]

        # Do we want to interpolate the data
        if bool_interp:
            data_interp = data_processing.interpolate_data(data=data['data'],
                                             num_frames=num_frames,
                                             num_frames_interp=params.num_frames_interp,
                                             num_features=len(features))
            samples_balanced = (data_interp, data['labels'])
            num_frames = params.num_frames_interp
        else:
            samples_balanced = (data['data'], data['labels'])

        # Split the data into training and test data before normalising and sectioning
        X_train, X_test, y_train, y_test = train_test_split(samples_balanced[0],
                                                            samples_balanced[1],
                                                            test_size=per_test,
                                                            random_state=params.random_state)

        # # Normalise the highD data across all locations
        data_train, scaler = data_processing.normalise_data({'data':X_train,
                                                            'labels':y_train})
        data_test, scaler = data_processing.normalise_data({'data':X_test,
                                                           'labels':y_test},
                                                           scaler=scaler)

        # Extract the different classes for the classification task
        samples_train, samples_train_balanced = \
            data_processing.get_classification_data(data_train,
                                                    num_frames,
                                                    frame_rate,
                                                    seconds_per_bin,
                                                    bool_augmented_labels=False)
        samples_test, samples_test_balanced = \
            data_processing.get_classification_data(data_test,
                                                    num_frames,
                                                    frame_rate,
                                                    seconds_per_bin,
                                                    bool_augmented_labels=False)

        # After balancing and sectioning, we want to overwrite the training and testing samples
        X_train, y_train = samples_train_balanced[0], samples_train_balanced[1]
        X_test, y_test = samples_test_balanced[0], samples_test_balanced[1]

        if bool_save_training_data:
            # Save the split training data
            file_name, extension = file_name.split('.')
            data_in = (X_train, X_test, y_train, y_test)
            data_processing.save_training_test(data_in,
                                               dir_name=params.data_path,
                                               file_name=file_name +
                                                         '_balanced_spb{}.'.format(seconds_per_bin)
                                                         + extension)

    # Extract the training data
    if 'balanced' in file_name:
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

    # Select only the features we want to train with
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

    # Define a line break
    line_break = '+' * 100

    print(line_break)
    print('Feature: {}'.format(features))
    print(line_break)

    ############################################################################
    ## Build the TTLC Classifiers                                             ##
    ############################################################################
    # Create a StratifiedKFold object to split the training data
    if params.kFolds > 1:
        skf = StratifiedKFold(n_splits=params.kFolds,
                              shuffle=True,
                              random_state=params.random_state)
    else:
        skf = StratifiedShuffleSplit(n_splits=params.kFolds,
                                     test_size=per_val,
                                     random_state=params.random_state)

    ## Define callback from TORCH HELPER
    # First create a folder for the call backs
    if bool_save and params.check_point:
        cb_path_epochs = os.path.sep.join([dir_name, 'epochs'])
        if not os.path.exists(cb_path_epochs):
            os.mkdir(cb_path_epochs)

    # Define empty dictionaries to store the results
    histories = dict()
    histories_cb = dict()
    models = dict()
    for model_name, model_arguments in params.models.items():
        # Use k-folds validation to get a better estimate of the
        # validation accuracy
        i = 0
        for train_ind, val_ind in skf.split(X_train, y_train):

            print(line_break)
            print('Training: ' + model_name)
            print(line_break)

            # Create the model
            model_tmp = ttlc.cnnTTLC(name=model_name +
                                    '_kFold{}'.format(i),
                                     cnnType=model_arguments[1],
                                     **model_arguments[0]).double()

            # Device conversion
            model_tmp.to(config.device)

            # Initialise the weights according to the initialisation we want
            initializer = th.WeightInitializer(params.initialiser,
                                               initializer_args=params.initialiser_args)
            model_tmp.apply(initializer.initialize_weights)

            # Initialise the optimiser
            optimiser = params.optimiser(model_tmp.parameters(),
                                         lr=params.lr)

            # Print the total number of parameters for each model
            print(model_name + ': {}'.format(th.nof_trainable_parameters(
                model_tmp)))
            # Also plot the model
            if params.verbose > 1:
                print(model_tmp)

            # Only create the dataloaders and callbacks if we want to train
            # the model
            if bool_train:
                # Reshape the training data if necessary
                if 'dense' in model_name:
                    X_train_tmp = X_train.reshape(X_train.shape[0], -1)
                else:
                    X_train_tmp = X_train

                # create a dataloader for the training and validation data for
                # this split
                train_loader = th.DataLoader(
                    data=(X_train_tmp[train_ind,].to(config.device),
                          y_train[train_ind,].to(config.device),
                          torch.arange(train_ind.shape[0]).to(config.device)),
                    batch_size=params.batch_size,
                    shuffle=True,
                    device=config.device
                )
                val_loader = th.DataLoader(
                    data=(X_train_tmp[val_ind,].to(config.device),
                          y_train[val_ind,].to(config.device),
                          torch.arange(val_ind.shape[0]).to(config.device)),
                    batch_size=params.batch_size,
                    shuffle=True,
                    device=config.device
                )

                # Define callbacks
                callbacks = list()
                # Add the History callback
                hist_callback = th.History(len(train_loader))
                callbacks.append(hist_callback)
                if params.early_stopping:
                    # Create a callback for early stopping
                    earlyStopper = th.EarlyStopping(
                        n_epoch_early_stop=params.early_stopping_patience,
                        verbose=True)
                    callbacks.append(earlyStopper)

                # Add the callback to save the
                if params.check_point and bool_save:
                    # Create the callback to store the weights
                    cb_path = os.path.sep.join([cb_path_epochs,
                                                'weights_{}_kFold{}'.format(
                                                    model_name, i) +
                                                '_{epoch:03d}.hdf5'])

                # Add the callback to calculate the cluster quality every few
                # epochs
                if params.cluster_per_epoch:
                    if bool_save:
                        cluster_dir = os.path.join(dir_name, 'clustering',
                                                   model_name)
                        utils.check_dir(cluster_dir)
                    else:
                        cluster_dir = None
                    featureCluster = th.FeatureClustering(train_loader=train_loader,
                                                          num_epochs=params.feature_cluster_epochs,
                                                          save_dir=cluster_dir)
                    callbacks.append(featureCluster)

                # Add a memory tracker
                # callbacks.append(th.PrintMemoryUsage())

                # Depending on the loss function, we need to pass a different
                # compute_loss function
                if params.loss == torch.nn.NLLLoss or params.loss == \
                        torch.nn.CrossEntropyLoss:
                    compute_loss = th.compute_cross_entropy_y_true_y_pred()

                history_tmp = th.train(model=model_tmp,
                                       epochs=params.epochs,
                                       train_loader=train_loader,
                                       optimizer=optimiser,
                                       compute_loss=compute_loss,
                                       val_loader=val_loader,
                                       callbacks=callbacks,
                                       verbose=params.verbose)

                if bool_save:
                    # Create path variable to save the model
                    PATH = os.path.join(path_weights,
                                        file_save_name + '_' + model_tmp.name + '.pt')
                    torch.save(model_tmp,
                                PATH)

            else:
                history_tmp = list()
                hist_callback = list()
            # Store the history and the model
            histories[model_name + '_kFolds{}'.format(i)] = history_tmp
            histories_cb[model_name + '_kFolds{}'.format(i)] = hist_callback
            models[model_name + '_kFolds{}'.format(i)] = model_tmp

            # Increment the kFolds counter
            i += 1

    if bool_save:
        eval_dict = dict()
        for model_name, model_tmp in models.items():
            # Evaluate models and save the test rates
            # Reshape the testing data if neccessary
            if 'dense' in model_name:
                X_test_tmp = X_test.reshape(X_test.shape[0], -1)
            else:
                X_test_tmp = X_test

            # Evaluate the models
            eval_tmp = th.evaluate_acc(model_tmp,
                                       X_test_tmp.to(config.device),
                                       y_test.to(config.device))
            eval_dict[model_name] = eval_tmp

        # Save the evaluations
        PATH = os.path.sep.join([dir_name, 'eval.pkl'])
        with open(PATH, 'wb') as file:
            pkl.dump(eval_dict, file)

        PATH = os.path.sep.join([dir_name, 'eval.txt'])
        with open(PATH, 'w') as file:
            for name, rate in eval_dict.items():
                file.write(name + ' : {:.3f}% \n'.format(rate*100))

        for model_name, history in histories.items():
            # Plot and save the learning curve
            PATH = os.path.join(path_history, model_name + '_history.png')
            plot_history(history,
                         filename=PATH)

        PATH = os.path.sep.join([dir_name, 'history.pkl'])
        with open(PATH, 'wb') as file:
            pkl.dump(histories, file)

        # Display the test accuracy
        print(line_break)
        print('Evaluate on Test:')
        print(line_break)
        for model_name, eval_tmp in eval_dict.items():
            print(model_name + ' : {:0.4f}%'.format(eval_tmp*100))
        print(line_break)

if __name__ == '__main__':
    # Run the script
    main()
