# Pyceptron
Python library for implementing a multilayer feedforward perceptron for arbitrary inputs and outputs

## Basic usage:

### import pyceptron
`import pyceptron as pct`

### create a model
first argument is a list of layer dimensions (at least two values) second argument is corresponding nonlinearities.
`model = pct.MLP([128,3,1], [None,'tanh','tanh'])`

### train the model
specify feature set, label set, learning rate and integer number of epochs
`result = model.fit(data, label, learning_rate=0.01, epochs=500)`
