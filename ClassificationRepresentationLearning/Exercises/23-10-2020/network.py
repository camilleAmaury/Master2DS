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
        return lasagne.layers.DenseLayer(hidden_layer_5, num_units=10, nonlinearity=softmax)