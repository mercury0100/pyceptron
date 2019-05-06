'''
Pyceptron.py - a modular multi-layer feed-forward neural network library
Arbitrary number of hidden layers with arbitrary number of neurons
tanh, logistic and relu activation with an optional softmax output layer
Weight decay, SGD momentum, mini-batch descent, dropout and cross-entropy loss are all available
Author: Cooper Doyle
'''
import numpy as np

class Activation(object):

    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        # a = np.tanh(x)
        return 1.0 - a**2
    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_deriv(self, a):
        # a = logistic(x)
        return  a * (1 - a )

    def __relu(self,x,alpha=0.):
        return np.where(x>=0, x, 0.)

    def __relu_deriv(self,a,alpha=0.):
        # a = relu(x)
        return np.where(a > 0, 1., 0.001)

    def __init__(self,activation='tanh'):
        if activation == 'logistic':
            self.f = self.__logistic
            self.f_deriv = self.__logistic_deriv
        elif activation == 'tanh':
            self.f = self.__tanh
            self.f_deriv = self.__tanh_deriv
        elif activation == 'relu':
            self.f = self.__relu
            self.f_deriv = self.__relu_deriv

class HiddenLayer(object):
    def __init__(self,n_in, n_out,
                 activation_last_layer='tanh',activation='tanh', W=None, b=None):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input=None
        self.activation=Activation(activation).f

        # activation deriv of last layer
        self.activation_deriv=None
        if activation_last_layer:
            self.activation_deriv=Activation(activation_last_layer).f_deriv

        self.W = np.random.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)
        )
        if activation == 'logistic':
            self.W *= 4

        # for relu, initialise weights in linear region
        if activation == 'relu':
            self.W = abs(np.random.randn(n_in,n_out)*np.sqrt(2/n_out))

        self.b = np.zeros(n_out,)

        #momentum term
        self.Vp = np.zeros(self.W.shape)
        self.V = np.zeros(self.W.shape)


        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def forward(self, input, dropout=None):
        '''
        :type input: numpy.array
        :param input: a symbolic tensor of shape (n_in,)
        '''
        if dropout:
            indices = np.random.choice(np.arange(input.size), replace=False, size=int(input.size * dropout))
            input[indices] = 0
        lin_output = np.dot(input, self.W) + self.b
        self.output = (
            lin_output if self.activation is None
            else self.activation(lin_output)
        )
        self.input=input
        return self.output

    def backward(self, delta, output_layer=False):
        self.grad_W = np.atleast_2d(self.input).T.dot(np.atleast_2d(delta))
        self.grad_b = delta
        if self.activation_deriv:
            delta = delta.dot(self.W.T) * self.activation_deriv(self.input)
        return delta

def get_mini_batches(X, y, batch_size):
    rand_ids = np.random.choice(len(y), len(y), replace=False)
    X_shuffled = X[rand_ids]
    y_shuffled = np.array(y)[rand_ids.astype(int)]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for i in range(0, len(y), batch_size)]
    return mini_batches

def soft_max(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def cross_entropy(y, y_hat, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: y_hat (N, k) ndarray
           y (N, k) ndarray
    Returns: scalar
    """
    y_hat = np.clip(y_hat, epsilon, 1. - epsilon)
    N = y_hat.shape[0]
    ce = -np.sum(y*np.log(y_hat+1e-9))/N
    return ce

class MLP:
    """
    """
    def __init__(self, layers, activation=[None,'tanh','tanh']):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic", "relu", "elu" or "tanh"
        :param weight_decay: The weight decay constant, between 0 and 1
        :param momentum: The SGD momentum constant, between 0 and 1
        """
        ### initialize layers
        self.layers=[]
        self.params=[]
        self.activation=activation
        for i in range(len(layers)-1):
            self.layers.append(HiddenLayer(layers[i],layers[i+1],activation[i],activation[i+1]))

    def forward(self,input,dropout=None,softmax=False):
        for layer in self.layers:
            output=layer.forward(input, dropout=dropout)
            input=output
        if softmax:
            output = soft_max(output)
        return output

    def criterion_MSE(self,y,y_hat,xent=False):
        activation_deriv=Activation(self.activation[-1]).f_deriv
        # MSE
        error = y-y_hat
        if not xent:
            loss = 0.5*sum(error**2)
        else:
            loss = cross_entropy(y,y_hat)
        # calculate the delta of the output layer
        delta=-error*activation_deriv(y_hat)
        # return loss and delta
        return loss,delta

    def backward(self,delta):
        delta=self.layers[-1].backward(delta,output_layer=True)
        for layer in reversed(self.layers[:-1]):
            delta=layer.backward(delta)

    def update(self,lr,momentum=0.,weight_decay=0.):
        for layer in self.layers:
            V = momentum*layer.Vp - lr*weight_decay*layer.W - lr*layer.grad_W
            layer.Vp = layer.V
            layer.V = V
            layer.W += layer.V
            layer.b -= lr * layer.grad_b

    def get_grads(self):
        layer_grad_W=[]
        layer_grad_b=[]
        for j in range(len(self.layers)):
            layer_grad_W.append(self.layers[j].grad_W)
            layer_grad_b.append(self.layers[j].grad_b)
        return layer_grad_W, layer_grad_b

    def batch_update(self,dW,db,lr,momentum=0.,weight_decay=0.):
        for j in range(len(self.layers)):
            V = momentum*self.layers[j].Vp - lr*weight_decay*self.layers[j].W - lr*dW[j]
            self.layers[j].Vp = self.layers[j].V
            self.layers[j].V = V
            self.layers[j].W += self.layers[j].V
            self.layers[j].b -= lr * db[j]

    def fit(self,X,y,learning_rate=0.1, momentum=0., weight_decay=0., epochs=100, batch_size=8, dropout=None, softmax=False, xent=False):
        """
        Online learning.
        :param X: Input data or features
        :param y: Input targets
        :param learning_rate: parameters defining the speed of learning
        :param momentum: sets the momentum coefficient for SGD, between 0 and 1
        :oaram weight_decay: sets the rate of weight decay between iterations, between 0 and 1
        :param epochs: number of times the dataset is presented to the network for learning
        :param batch_size: selects the batch size for mini-batch gradient descent. if 'None' then a more efficient loop is used for SGD.
            NB: Stochastic gradient descent only works for output_activation = 'logistic'
        :param dropout: sets the rate of dropout in the network during forward propagation.
        :param softmax: Adds a softmax layer to the output
        :param Xent: Employs cross-entropy loss function
        """
        X=np.array(X)
        y=np.array(y)
        to_return = np.zeros(epochs)

        # Stochastic gradient descent loop
        if not batch_size:
            for k in range(epochs):
                loss=np.zeros(X.shape[0])
                for it in range(X.shape[0]):
                    i=np.random.randint(X.shape[0])
                    # forward pass
                    y_hat = self.forward(X[i], dropout=dropout, softmax=softmax)
                    # backward pass
                    loss[it],delta=self.criterion_MSE(y[i],y_hat,xent=xent)
                    self.backward(delta)
                    # update
                    self.update(learning_rate, momentum, weight_decay)
                    to_return[k] = np.mean(loss)

        # Mini-batch gradient descent loop
        else:
            for k in range(epochs):
                batches = get_mini_batches(X,y,batch_size)
                batch_loss = np.zeros(len(batches))
                b = 0 # track batch number
                for batch in batches:
                    Xs = np.array(batch[0])
                    Ys = np.array(batch[1])
                    loss=np.zeros(Xs.shape[0])
                    dW, db = [], []
                    for i in range(Xs.shape[0]):
                        # forward pass
                        y_hat = self.forward(Xs[i], dropout=dropout, softmax=softmax)
                        # backward pass
                        loss[i],delta=self.criterion_MSE(Ys[i],y_hat,xent=xent)
                        self.backward(delta)
                        # fetch and store gradients
                        layer_grad_W, layer_grad_b = self.get_grads()
                        dW.append(layer_grad_W)
                        db.append(layer_grad_b)
                    # obtain mean batch loss
                    batch_loss[b] = np.mean(loss)
                    b += 1 # batch number increment
                    # calculate average batch gradients
                    gradients_W = {}
                    gradients_b = {}
                    for i in range(len(dW[0])): #could replace with len(self.layers)
                        gradients_W[i] = np.array([j[i] for j in dW]).mean(axis=0)
                        gradients_b[i] = np.array([j[i] for j in db]).mean(axis=0)
                    DW=[i for j,i in gradients_W.items()]
                    Db=[i for j,i in gradients_b.items()]
                    # update weights with batch gradient
                    self.batch_update(DW, Db, learning_rate, momentum, weight_decay)
                to_return[k] = np.mean(batch_loss)
        return to_return

    def predict(self, x):
        x = np.array(x)
        output = []
        for i in np.arange(x.shape[0]):
            ls = self.forward(x[i,:]).tolist()
            max_value=max(ls)
            max_index = ls.index(max_value)
            array = np.zeros(10)
            array[max_index] = 1
            output.append(array)
        np.array(output)
        return output
