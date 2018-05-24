import numpy as np
from math import *
import random
from datetime import datetime
import copy


train_set={(1,1):1,
           (1,0):0,
           (0,1):0,
           (0,0):0}

class Perceptron:

    def __init__(self, inputs, l_rate=0.05):
        assert type(inputs)==int
        self.weights=2*np.random.rand(inputs+1)-1
        self.l_rate=l_rate

    def __call__(self, inputs):
        assert len(inputs)==len(self.weights)-1
        inputs=np.array(inputs)
        sum=0
        inputs=np.concatenate((inputs, np.array([1])), axis=0)
        for weight, input in zip(self.weights, inputs):
            sum+=input*weight
        #print(inputs[0], self.weights, 1/(1+exp(-sum)))
        return 1/(1+exp(-sum))

    """def __call__(self, inputs):
        assert len(inputs)==len(self.weights)-1
        inputs=np.array(inputs)
        sum=0
        inputs=np.concatenate((inputs, np.array([1])), axis=0)
        for weight, input in zip(self.weights, inputs):
            sum+=input*weight
        #print(inputs[0], self.weights, sum)
        return sum"""

    def train(self, training_set):
        error=0
        for inputs in training_set.keys():
            expectation=training_set[inputs]
            output=self(inputs)
            inputs=np.concatenate((inputs, np.array([1])), axis=0)
            for i in range(len(inputs)):
                self.weights[i]=self.weights[i]+self.l_rate*(expectation-output)*inputs[i]
            error+=output-expectation
        return error


class Neuron(Perceptron):

    def __init__(self, input_layer, l_rate=0.1):
        super().__init__(len(input_layer), l_rate)
        self.input_layer=input_layer
        self.output=None

    def __call__(self, inputs):
        output=super().__call__(inputs)
        self.output=output
        return output

    def deriv(self, x):
        return exp(x)/(1+exp(x))**2

    def train(self, weighed_deltas):
        #print("output", self.output)
        #print("weighed_deltas", weighed_deltas)
        my_delta=sum(self.output*(1-self.output)*weighed_deltas)
        #print(1, my_delta)
        prev_outputs=self.input_layer.output_cache
        #print(2, prev_outputs)
        #print(3, self.weights)
        self.weights-=self.l_rate*np.concatenate((prev_outputs, np.array([1])), axis=0)*my_delta
        #print(4, self.weights)
        return my_delta*self.weights


class Layer:
    def __init__(self, input_layer, nodes, l_rate=0.1):
        assert isinstance(input_layer, Layer)
        self.neurons=[]
        for i in range(nodes):
            self.neurons.append(Neuron(input_layer, l_rate))
        self.output_cache=None

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

class Network:
    def __init__(self, inputs, l_rate=0.1):
        assert type(inputs)==int
        self.inputs=inputs
        self.layers=[InputLayer(inputs)]
        self.l_rate=l_rate

    def add_layer(self, nodes):
        self.layers.append(Layer(self.layers[len(self.layers)-1], nodes, l_rate=self.l_rate))
        return self

    def __call__(self, inputs):
        inputs=np.array(inputs)
        outputs=inputs
        for layer in self.layers:
            outputs=layer(outputs)
        #print("final output", outputs)
        return outputs

    def evaluate(self, inputs, expectations):
        error=0
        outputs=self(np.array(inputs))
        for diff in expectations-outputs:
            error+=diff*diff
        return error/len(outputs)

    def backprop(self, inputs, expectations):
        outputs=self(inputs)
        weighed_delta=np.array([(outputs-np.array(expectations))/len(outputs)])
        #print("first delta", weighed_delta)
        for i in range(1,len(self.layers)):
            weighed_delta=np.array(self.layers[-i].backprop(weighed_delta))

        error= self.evaluate(inputs, expectations)
        #print()
        #print()
        return error

    def train(self, data):
        assert isinstance(data, dict)
        for key in data:
            self.backprop(np.array(key), np.array(data[key]))
        return 1



if __name__=="__main__":
    nn=Network(2, l_rate=i*0.05)
    nn.add_layer(4)
    nn.add_layer(4)
    nn.add_layer(4)
    nn.add_layer(1)
    for i in range(1,8,1):
        start=datetime.now()
        nntest=copy.copy(nn)
        for i in range(5000):
            nntest.train(train_set)
            if i%1000==0:
                print("generation", i, ":", (datetime.now()-start).seconds+(datetime.now()-start).microseconds/10**6,"seconds elapsed")
        print(nntest((0,0)))
        print(nntest((0,1)))
        print(nntest((1,0)))
        print(nntest((1,1)))
