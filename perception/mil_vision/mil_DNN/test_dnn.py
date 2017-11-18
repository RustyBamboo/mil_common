#!/usr/bin/env python
import pandas as pd
import rospy
import numpy as np
from layers.fully_connected import FullyConnected
from NNet import NNet
import matplotlib.pyplot as plt
import matplotlib.cm as cm




data = pd.read_csv('HW6_Data.txt', header=None, sep='\t')
data = np.array(data)

class Sigmoid(object):
    def compute(self, x):
        return 1. / (1. + np.exp(-x))

    def deriv(self, x):
        y = self.compute(x)
        return y * (1. - y)

class Tanh(object):
    def compute(self, x):
        return np.tanh(x)

    def deriv(self, x):
        # print x, 1.0-x**2
        return 1.0 - x**2

class RELU(object):
    def compute(self, x):
        return np.maximum(0, x)
    def deriv(self, x):
        return 1. * (x > 0)

class MeanSquaredError(object):
    def compute(self, (X, Y)):
        return (1. / 2. * X.shape[0]) * ((X - Y) ** 2.)

    def deriv(self, (X, Y)):
        # return (X - Y)
        return (Y-X)

s = Sigmoid()
t = Tanh()
r = RELU()
mse = MeanSquaredError()

l1 = FullyConnected((2,2), activation=t, weight_init=lambda shp: np.random.normal(size=shp), bias=True)
l1.W = np.array([[0, 0.1, 0.2],
                [0, 0.0927, -1.0133], 
                [0, -1.0533, -0.5942]])
# l1.W = np.array([[0.0927, -1.0133], [-1.0533, -0.5942]])
l2 = FullyConnected((2,1), activation=t, weight_init=lambda shp: np.random.normal(size=shp), bias=True)
l2.W = np.array([[0.3],
                [-0.9057],
                [0.2664]])
# l2.W = np.array([[-0.9057], [0.2664]])

layers = [l1, l2]
net = NNet(layers, 0.1, mse)


for i in range(5000):
    if (i % 1000 == 0): print i
    i = np.random.randint(data[:,0:2].shape[0])
    net.train_step((np.insert(data[i,0:2],0,1), data[i,2]))

print l1.W
print l2.W

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
xx,yy= np.meshgrid(x, y)
positions = np.c_[xx.ravel(), yy.ravel()]
positions = np.hstack((np.ones((positions.shape[0],1)), positions))

Z = np.array([net.forward_propagation(p) for p in positions])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z)
plt.scatter(data[:,0], data[:,1], c = cm.hot(data[:,2]))

plt.show()