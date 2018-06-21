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

        self.__l_rate=l_rate
        self.__function=function
        self.ATTR={"function":self.__function, "l_rate":self.__l_rate}

    def set_attr(self, attrname, value):
        for i in range(len(self.neurons)):
            self.neurons[i].set_attr(attrname, value)

        if attrname=="function":
            self.function=value
        elif attrname=="l_rate":
            self.l_rate=l_rate
        else:
            raise Exception("Wrong key")

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
