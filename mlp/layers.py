
# Machine Learning Practical (INFR11119),
# Pawel Swietojanski, University of Edinburgh

import numpy
import logging
from mlp.costs import Cost
from scipy.special import expit

logger = logging.getLogger(__name__)


class MLP(object):
    """
    This is a container for an arbitrary sequence of other transforms
    On top of this, the class also keeps the state of the model, i.e.
    the result of forward (activations) and backward (deltas) passes
    through the model (for a mini-batch), which is required to compute
    the gradients for the parameters
    """
    def __init__(self, cost):

        assert isinstance(cost, Cost), (
            "Cost needs to be of type mlp.costs.Cost, got %s" % type(cost)
        )

        self.layers = [] #the actual list of network layers
        self.activations = [] #keeps forward-pass activations (h from equations)
                              # for a given minibatch (or features at 0th index)
        self.deltas = [] #keeps back-propagated error signals (deltas from equations)
                         # for a given minibatch and each layer
        self.cost = cost

    def fprop(self, x):
        """

        :param inputs: mini-batch of data-points x
        :return: y (top layer activation) which is an estimate of y given x
        """

        if len(self.activations) != len(self.layers) + 1:
            self.activations = [None]*(len(self.layers) + 1)

        self.activations[0] = x
        for i in xrange(0, len(self.layers)):
            self.activations[i+1] = self.layers[i].fprop(self.activations[i])
        return self.activations[-1]

    def bprop(self, cost_grad):
        """
        :param cost_grad: matrix -- grad of the cost w.r.t y
        :return: None, the deltas are kept in the model
        """

        # allocate the list of deltas for each layer
        # note, we do not use all of those fields but
        # want to keep it aligned 1:1 with activations,
        # which will simplify indexing later on when
        # computing grads w.r.t parameters
        if len(self.deltas) != len(self.activations):
            self.deltas = [None]*len(self.activations)

        # treat the top layer in special way, as it deals with the
        # cost, which may lead to some simplifications
        top_layer_idx = len(self.layers)
        self.deltas[top_layer_idx], ograds = self.layers[top_layer_idx - 1].\
            bprop_cost(self.activations[top_layer_idx], cost_grad, self.cost)

        # then back-prop through remaining layers
        for i in xrange(top_layer_idx - 1, 0, -1):
            self.deltas[i], ograds = self.layers[i - 1].\
                bprop(self.activations[i], ograds)

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_layers(self, layers):
        self.layers = layers

    def get_name(self):
        return 'mlp'


class Layer(object):
    """
    Abstract class defining an interface for
    other transforms.
    """
    def __init__(self, rng=None):

        if rng is None:
            seed=[2015, 10, 1]
            self.rng = numpy.random.RandomState(seed)
        else:
            self.rng = rng

    def fprop(self, inputs):
        """
        Implements a forward propagation through the i-th layer, that is
        some form of:
           a^i = xW^i + b^i
           h^i = f^i(a^i)
        with f^i, W^i, b^i denoting a non-linearity, weight matrix and
        biases at the i-th layer, respectively and x denoting inputs.

        :param inputs: matrix of features (x) or the output of the previous layer h^{i-1}
        :return: h^i, matrix of transformed by layer features
        """
        raise NotImplementedError()
    
    def bprop(self, h, igrads):
        """
        N.B. Using the notation of res/code_scheme.svg
        * h^i = f(a^i)            : the output of layer i (f is some function) [1 x K^i vector]
        * x   = h^{i-1}           : the input to layer i (h^0 = first inputs) [1 x K^{i-1} vector]
        * a^i = x W^i + b^i       : the activation of the layer [1 x K^i vector]
        * d^i = g^{i+1} dh^i/da^i : \delta for layer i [1 x K^i vector]
        * g^i = d^i (W^i).T       : the 'grads' for layer i [1 x K^i vector]
        
        Implements a backward propagation from a generic layer i+1 to layer i 
        i.e. we calculate deltas = d^i and ograds = g^i. The deltas will be
        used to calculate the update gradient of the Error w.r.t. W^i and b^i:
            dE/dW^i = h^{i-1} d^i
            dE/dW^i = h^{i-1} d^i
        The grads propagate the error back to layer i-1 i.e. the ograds from this
        call are used as igrads for the next.

        :param h      : h^i - Used to create the function dh^i/da^i
                        ***even if unused, must be present for consistency amongst layers***
        :param igrads : g^{i+1} - Used to calculate deltas
            
        :return: a tuple (deltas, ograds) where:
            deltas = d^i = igrads  * dh^i/da^i
                         = g^{i+1} * dh^i/da^i
            ograds = g^i = dh^i/dx^i
                         = dh^i/da^i da^i/dx^i
                         = d^i (W^i).T
        
        N.B. da^i/dx^i = d/dx^i (xW^i + b^i) = (W^i).T
        """
        raise NotImplementedError()

    def bprop_cost(self, h, igrads, cost=None):
        """
        Implements a backward propagation in case the layer directly
        deals with the optimised cost (i.e. the top layer)
        By default, method should implement a back-prop for default cost, that is
        the one that is natural to the layer's output, i.e.:
        linear -> mse, softmax -> cross-entropy, sigmoid -> binary cross-entropy
        :param h: it's an activation produced in forward pass
        :param igrads, error signal (or gradient) flowing to the layer, note,
               this in general case does not corresponds to 'deltas' used to update
               the layer's parameters, to get deltas ones need to multiply it with
               the dh^i/da^i derivative
        :return: a tuple (deltas, ograds) where:
               deltas = igrads * dh^i/da^i
               ograds = deltas \times da^i/dx^i
        """

        raise NotImplementedError()

    def pgrads(self, inputs, deltas):
        """
        Return gradients w.r.t parameters
        """
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def set_params(self):
        raise NotImplementedError()

    def get_name(self):
        return 'abstract_layer'


class Linear(Layer):

    def __init__(self, idim, odim,
                 rng=None,
                 irange=0.1):

        super(Linear, self).__init__(rng=rng)

        self.idim = idim
        self.odim = odim

        self.W = self.rng.uniform(
            -irange, irange,
            (self.idim, self.odim))

        self.b = numpy.zeros((self.odim,), dtype=numpy.float32)
    

    def get_name(self):
        return 'linear'
    
    def fprop(self, inputs):
        a = numpy.dot(inputs, self.W) + self.b
        # here f() is an identity function, so just return a linear transformation
        return a

    def bprop(self, h, igrads):
        # since df^i/da^i = 1 (f is assumed identity function),
        # deltas are in fact the same as igrads
        ograds = numpy.dot(igrads, self.W.T)
        return igrads, ograds

    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'mse':
            # for linear layer and mean square error cost,
            # cost back-prop is the same as standard back-prop
            return self.bprop(h, igrads)
        else:
            raise NotImplementedError('Linear.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())

    def pgrads(self, inputs, deltas):
        grad_W = numpy.dot(inputs.T, deltas)
        grad_b = numpy.sum(deltas, axis=0)

        return [grad_W, grad_b]

    def get_params(self):
        return [self.W, self.b]

    def set_params(self, params):
        #we do not make checks here, but the order on the list
        #is assumed to be exactly the same as get_params() returns
        self.W = params[0]
        self.b = params[1]

        
class Sigmoid(Linear):
    
    def get_name(self):
        return 'sigmoid'
    
    ## Showing I could roll my own
    # def sigmoid(self, X):
    #     return 1. / (1 + numpy.exp(-X))
    # expit is 10x faster http://stackoverflow.com/questions/21106134/numpy-pure-functions-for-performance-caching
    def sigmoid(self, X):
        return expit(X)
    
    def fprop(self, inputs):
        a = super(Sigmoid, self).fprop(inputs)
        return self.sigmoid(a)
    
    def bprop(self, h, igrads):
        # h = Sigmoid(a) = 1. / (1 + numpy.exp(-a))
        # dh/da = numpy.exp(-a) / (1 + numpy.exp(-a))**2
        #       = h(1-h)
        deltas = igrads * h * (1-h)
        ograds = deltas.dot(self.W.T)
        return deltas, ograds
    
    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'ce':
            # for Sigmoid layer and cross entropy cost,
            # cost back-prop is the same as standard back-prop
            return self.bprop(h, igrads)
        else:
            raise NotImplementedError('Linear.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())


class Softmax(Linear):
    def get_name(self):
        return 'softmax'
    
    def softmax(self, X):
        ## For matrices
        ex = numpy.exp(X)
        # Hack for a batch of size 1...must be a better way to deal with this
        if ex.ndim == 1:
            ex = ex[numpy.newaxis, :]
        tot = numpy.sum(ex, axis=1, keepdims=True)
        assert(tot.shape[0] == ex.shape[0]), \
            "Total of exponents should be size N. Sum size %d, N from X is %d" % (tot.shape[0], ex.shape[0])
        return ex / tot
#         ex = numpy.exp(x)
#         tot = numpy.sum(ex)
#         return ex / tot
    
    def fprop(self, inputs):
        a = super(Softmax, self).fprop(inputs)
        return self.softmax(a)
    
    # TODO: should probably NotImplementedError bprop and edit bprop_cost
    def bprop(self, h, igrads):
        # h = Softmax(a) = np.exp(a) / np.sum(np.exp(a))
        # dh_c/da_k = h_c(\dirac_ck - h_k)  #NB a matrix
        # see end of lecture notes mlp05-hid
        # USE AS TOP LAYER ONLY
        ograds = numpy.dot(igrads, self.W.T)
        return igrads, ograds
        
    def bprop_cost(self, h, igrads, cost):
        if cost is None or cost.get_name() == 'ce':
            # for softmax layer and cross entropy cost,
            # cost back-prop is the same as standard back-prop
            return self.bprop(h, igrads)
        else:
            raise NotImplementedError('Linear.bprop_cost method not implemented '
                                      'for the %s cost' % cost.get_name())
        return deltas, ograds


        