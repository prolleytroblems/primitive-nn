import numpy as np
from math import *
from neuron import *


class Layer:
    def __init__(self, input_layer, nodes, l_rate=0.1, function="logistic"):
        assert isinstance(input_layer, Layer)
        self.neurons=[]
        for i in range(nodes):
            self.neurons.append(Neuron(input_layer, l_rate, function=function))
        self.output_cache=None
        self.function=function

    def __call__(self, inputs):
        assert isinstance(inputs, np.ndarray)
        output=[]
        for neuron in self.neurons:
            output.append(neuron(inputs))
        self.output_cache=np.array(output)
        return self.output_cache

    def __len__(self):
        return len(self.neurons)

    def backprop(self, weighed_deltas):
        my_w_deltas=[]
        for i, neuron in enumerate(self.neurons):
            my_w_deltas.append(neuron.train(weighed_deltas[:,i]))
        return np.array(my_w_deltas)

class InputLayer(Layer):
    def __init__(self, inputs):
        self.inputs=inputs
        self.output_cache=None

    def __call__(self, inputs):
        assert isinstance(inputs, np.ndarray)
        self.output_cache=inputs
        return inputs

    def __len__(self):
        return self.inputs

    def backprop(self):
        raise Exception()

if __name__=="__main__":
    pass
