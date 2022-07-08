# TinyNN
A lightweight multilayer feedforward perceptron for arbitrary inputs and outputs built in numpy. In development:

1. ~ReLU activation~
2. ~Weight Decay~
3. ~Momentum in SGD~
4. ~Dropout~
5. ~Softmax and cross-entropy loss~
6. ~Mini-batch training~
7. Batch normalisation
8. Multiprocessing for mini-batch descent

## Basic usage:

### import tinynn
`import tinynn as tnn`

### create a model
first argument is a list of layer dimensions (at least two values) second argument is corresponding nonlinearities.

`model = tnn.MLP([128,3,1], [None,'tanh','tanh'])`

### train the model
specify feature set, label set, learning rate and integer number of epochs

`result = model.fit(data, label, learning_rate=0.01, epochs=500)`

### generate predictions
`model.predict(feature_data)`
