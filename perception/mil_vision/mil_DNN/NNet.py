import numpy as np
from collections import deque

class NNet(object):
    def __init__(self, layers, lrate, loss):
        self.layers = layers
        self.loss = loss
        self.lrate = lrate

    def forward_propagation(self, x_in):
        output = x_in
        for layer in self.layers:
            output = layer.propagate_forward(output)
        return output


    def train_step(self, mini_batch):
        mini_batch_inputs, mini_batch_outputs = mini_batch
        # print mini_batch_inputs
        zs = deque([mini_batch_inputs])
        activation = mini_batch_inputs
        # print self.layers
        for l in self.layers:
            z, activation = l.propagate_forward_2(activation)
            # print (z, activation)
            zs.appendleft(activation)

        # print "zs: ", zs
        loss_err = self.loss.deriv((activation, mini_batch_outputs))
        # assert False
        lz = zs.popleft()
        backwarded_err = loss_err
        grads = deque()
        for l in reversed(self.layers):
            layer_err = l.local_error(lz, backwarded_err) #local
            lz = zs.popleft()
            grads.appendleft(l.gradient(lz, layer_err))
            backwarded_err = l.backward_propagation(layer_err) # backwarded error

        # print "-"*50, "Update Step", "-"*50
        # update step
        for l in self.layers:
            l.update(self.lrate * grads.popleft())

        assert len(grads) == 0