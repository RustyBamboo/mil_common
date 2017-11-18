from abc import ABCMeta, abstractmethod

class Layer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def propagate_forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def propagate_forward_2(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def local_error(self, z, backwarded_err):
        raise NotImplementedError()

    @abstractmethod
    def backward_propagation(self, layer_err):
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, inputs, layer_err):
        raise NotImplementedError()

    @abstractmethod
    def update(self, grad):
        raise NotImplementedError()
