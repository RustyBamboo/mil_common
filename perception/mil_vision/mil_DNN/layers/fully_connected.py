
from layer import Layer

import numpy as np
class FullyConnected(Layer):
    def __init__(self, wshape, activation, weight_init, bias=False):
        self.wshape = wshape
        self.W = weight_init((self.wshape[0]+1, self.wshape[1]+1)) if bias else weight_init(self.wshape)
        self.activation = activation
        self.bias = bias
        print self.W
        self.debug = False

    def propagate_forward(self, inputs):
        return self.activation.compute(inputs.dot(self.W))

    def propagate_forward_2(self, inputs):

        inputs = np.atleast_2d(inputs)
        if self.debug:
            print "-" * 50, " TRAIN FORWARD ", "-" * 50
            print "Input", inputs
            print "W", self.W
            # print "W-modified", self.W[:,1:]
            # print self.wshape
        z = np.atleast_2d(inputs).dot(self.W)
        if self.debug:
            print "z", z
            print "z-act", self.activation.compute(z)
        return (z, self.activation.compute(z))

    def local_error(self, z, backwarded_err):
        act = self.activation.deriv(z)
        if self.debug:
            print "-" * 50, " GET LAYER ERROR", "-" * 50

            print "z",z
            print "back_err", backwarded_err

            print "z-derv", self.activation.deriv(z)
            print "act", act
            print "error", np.atleast_2d(backwarded_err) * act
        return np.atleast_2d(backwarded_err) * act

    def backward_propagation(self, layer_err):
        if self.debug:
            print "-" * 50, "BACKWARD", "-" * 50
            print "layer_err",layer_err
            print "C-W", self.W.T
            print "out", np.atleast_2d(layer_err).dot(self.W.T)
        return np.atleast_2d(layer_err).dot(self.W.T)

    def gradient(self, inputs, layer_err):
        x= np.atleast_2d(inputs).T.dot(np.atleast_2d(layer_err))
        if self.debug:
            print "-" * 50, "GET_GRAD", "-" * 50
            print "in", inputs.T
            print "layer_err", layer_err
            print x
        return x

    def update(self, grad):
        if self.debug:
            print "-" * 50, "GRAD", "-" * 50
            print "W", self.W
            print "Grad", grad
        self.W = self.W + grad