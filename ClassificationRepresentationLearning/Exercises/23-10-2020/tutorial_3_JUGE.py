#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------------------------------------------------#
# ------------------------------ Imports ----------------------------------------- #
# ---------------------------------------------------------------------------------#

from __future__ import print_function

import sys
import os
import time

import theano
import theano.tensor as T
import lasagne

import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 5, 10


# ---------------------------------------------------------------------------------#
# ------------------------------ Dataset ----------------------------------------- #
# ---------------------------------------------------------------------------------#

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print(">   Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------------#
# ---------------------------------- MLP ----------------------------------------- #
# ---------------------------------------------------------------------------------#

class MLP(object):
    
    def __init__(self, input_var, output_var, regularization, loss, from_file=None):
        # initializating network configuration
        self._create_network(input_var)
        if from_file != None:
            self.load_model(from_file)
        prediction = lasagne.layers.get_output(self.network)
        self.loss = loss(prediction, output_var)
        self.loss = self.loss.mean()
        self.loss += regularization[1] * lasagne.regularization.regularize_network_params(self.network, regularization[0])
        self.params = lasagne.layers.get_all_params(self.network, trainable=True)
        self.optimizer = lasagne.updates.adam(self.loss, self.params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        test_prediction = lasagne.layers.get_output(self.network, deterministic=True)
        self.test_loss = loss(test_prediction,output_var)
        self.test_loss = self.test_loss.mean()
        
        # As a bonus, also create an expression for the classification accuracy:
        train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), output_var), dtype=theano.config.floatX)
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), output_var), dtype=theano.config.floatX)
        # training_loss function
        self.train_fn = theano.function([input_var, output_var], self.loss, updates=self.optimizer)
        # validation loss and accuracy function
        self.val_fn = theano.function([input_var, output_var], [self.test_loss, test_acc])
    
    # Instanciate the neural network
    def _create_network(self, input_var):
        # ReLU activation function
        reLU = lasagne.nonlinearities.rectify
        # Softmax function
        softmax = lasagne.nonlinearities.softmax
        
        # input_layers
        input_layer = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
        drop_input_layer = lasagne.layers.dropout(input_layer, p=0.08)
        
        # hidden layers 1
        hidden_layer_1 = lasagne.layers.DenseLayer(drop_input_layer, num_units=400,
                nonlinearity=reLU, W=lasagne.init.GlorotUniform(gain='relu'))
        drop_hidden_layer_1 = lasagne.layers.dropout(hidden_layer_1, p=0.25)
        # hidden layers 2
        hidden_layer_2 = lasagne.layers.DenseLayer(drop_hidden_layer_1, num_units=350,
                nonlinearity=reLU, W=lasagne.init.GlorotUniform(gain='relu'))
        drop_hidden_layer_2 = lasagne.layers.dropout(hidden_layer_2, p=0.15)
        # hidden layers 3
        hidden_layer_3 = lasagne.layers.DenseLayer(drop_hidden_layer_2, num_units=300,
                nonlinearity=reLU, W=lasagne.init.GlorotUniform(gain='relu'))
        drop_hidden_layer_3 = lasagne.layers.dropout(hidden_layer_3, p=0.05)
        # hidden layers 4
        hidden_layer_4 = lasagne.layers.DenseLayer(drop_hidden_layer_3, num_units=250,
                nonlinearity=reLU, W=lasagne.init.GlorotUniform(gain='relu'))
        drop_hidden_layer_4 = lasagne.layers.dropout(hidden_layer_4, p=0.05)
        # hidden layers 5
        hidden_layer_5 = lasagne.layers.DenseLayer(drop_hidden_layer_4, num_units=200,
                nonlinearity=reLU, W=lasagne.init.GlorotUniform(gain='relu'))
        
        # output layer
        self.network = lasagne.layers.DenseLayer(hidden_layer_5, num_units=10, nonlinearity=softmax)
    

    # batch behavior function
    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
        
    # Training method
    def training(self, data, epochs, batch_size=500, verbose=True):
        evolution = []
        # iteration over the epoch
        for epoch in range(epochs):
            if verbose:
                start_time = time.time()
                
            # Training set 
            train_err = 0
            train_batches = 0
            for batch in self.iterate_minibatches(data[0], data[1], batch_size, True):
                inputs, targets = batch
                err = self.train_fn(inputs, targets)
                train_err += err
                train_batches += 1

            # Validation set
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(data[2], data[3], batch_size):
                inputs, targets = batch
                err, acc = self.val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1        
            
            # printing the epoch's result
            if verbose:
                evolution.append((train_err / train_batches, val_err / val_batches, val_acc / val_batches * 100))
                sys.stdout.write("\rEpoch {} of {} took {:.3f}s => t_l : {:.6f}; v_l : {:.6f}; v_a : {:.2f}%     ".format(epoch + 1, epochs, time.time() - start_time,
                                    evolution[epoch][0], evolution[epoch][1], evolution[epoch][2]))
                sys.stdout.flush()
        return evolution
    
    # Training method
    def evaluate(self, X, Y, batch_size=500, verbose=True):
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.iterate_minibatches(X, Y, batch_size):
            inputs, targets = batch
            err, acc = self.val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        evaluation = (test_err / test_batches, test_acc / test_batches * 100)
        if verbose:
            print("\nFinal results:")
            print("  test loss:\t\t\t{:.6f}".format(evaluation[0]))
            print("  test accuracy:\t\t{:.2f} %".format(evaluation[1]))
        return evaluation
    
    # serializing method
    def save_model(self, file):
        np.savez('{}.npz'.format(file), *lasagne.layers.get_all_param_values(self.network))
    
    # deserializing method
    def load_model(self, file):
        with np.load('{}.npz'.format(file)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.network, param_values)


# ---------------------------------------------------------------------------------#
# --------------------------------- Plot ----------------------------------------- #
# ---------------------------------------------------------------------------------#
def plotify(num_epochs, train_):
    
    # plot results
    range_ = np.arange(num_epochs)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(range_, [i[0] for i in train_])
    axs[0].set_title('Loss on training set during training')
    axs[1].plot(range_, [i[1] for i in train_], 'tab:orange')
    axs[1].set_title('Loss validation set during training')
    axs[2].plot(range_, [i[2] for i in train_], 'tab:green')
    axs[2].set_title('Accuracy on validation set during training')

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    fig.show()


# ---------------------------------------------------------------------------------#
# --------------------------------- Main ----------------------------------------- #
# ---------------------------------------------------------------------------------#

def main(model='mlp', from_file=None, num_epochs=50):
    # Load the dataset
    print(">   Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    data = [X_train, y_train, X_val, y_val]
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    output_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print(">   Building model and compiling functions...")
    if model == "mlp":
        model = MLP(input_var, output_var, [lasagne.regularization.l2, 0.0002], 
                    lasagne.objectives.categorical_crossentropy, from_file=from_file)
    elif model == "cnn":
        raise Exception("CNN not implemented yet")
    else:
        raise Exception("Choose a valid Model : 'mlp' or 'cnn'")
    
    train_ = model.training(data, num_epochs)
    
    plotify(num_epochs, train_)
    
    eval_ = model.evaluate(X_test, y_test)
    
    return model


# ---------------------------------------------------------------------------------#
# ---------------------------------- Run ----------------------------------------- #
# ---------------------------------------------------------------------------------#

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP)")
        print("HOW TO USE :")
        print('$python tutorial_3_JUGE.py "mlp" 50 "mlp_juge"')
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 3:
            kwargs['from_file'] = sys.argv[3]
        else:
            kwargs['from_file'] = None
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        model = main(**kwargs)
        model.save_model("mlp_juge")