import numpy as np
from math import *
import random
from datetime import datetime
import copy



class Perceptron:

    def __init__(self, inputs, l_rate=0.05, function="logistic"):
        assert type(inputs)==int
        self.weights=2*np.random.rand(inputs+1)-1
        self.l_rate=l_rate
        self.function=function

    def __call__(self, inputs):
        assert len(inputs)==len(self.weights)-1
        inputs=np.array(inputs)
        sum=0
        inputs=np.concatenate((inputs, np.array([1])), axis=0)
        for weight, input in zip(self.weights, inputs):
            sum+=input*weight
        return self.activationf(sum)

    def activationf(self, u):
        if self.function=="logistic":
            return 1/(1+exp(-u))
        elif self.function=="relu":
            if u<0:
                return 0
            else:
                return u
        elif self.function==None:
            return u
        else:
            raise Exception()

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
        my_delta=sum(self.output*(1-self.output)*weighed_deltas)
        prev_outputs=self.input_layer.output_cache
        self.weights-=self.l_rate*np.concatenate((prev_outputs, np.array([1])), axis=0)*my_delta
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
    def __init__(self, inputs, l_rate=0.1, type="regressor"):
        assert type(inputs)==int
        self.inputs=inputs
        self.layers=[InputLayer(inputs)]
        self.l_rate=l_rate
        self.__type=type

    def add_layer(self, nodes):
        self.layers.append(Layer(self.layers[len(self.layers)-1], nodes, l_rate=self.l_rate))
        return self

    def __call__(self, inputs):
        if self.__type=="regressor":
            inputs=np.array(inputs)
            outputs=inputs
            for layer in self.layers:
                outputs=layer(outputs)
            return outputs
        elif self.__type=="classifier":
            inputs=np.array(inputs)
            outputs=inputs
            for layer in self.layers:
                outputs=layer(outputs)
            exp_sum=sum([exp(output) for output in outputs])
            outputs=[exp(output)/exp_sum for output in outputs]
            return outputs
        else:
            raise Exception()

    def evaluate(self, dataset):
        if self.__type=="regressor":
            error=0
            for key in dataset:
                outputs=self(np.array(key))
                for diff in dataset[key]-outputs:
                    error+=diff*diff
            return error/len(dataset)
        elif self.__type=="classifier":
            mistakes=0
            for key in dataset:
                outputs=self(np.array(key))
                if np.where(outputs==max(outputs))!=np.where(np.array(dataset[key])==max(dataset[key])):
                    mistakes+=1
            return mistakes/len(dataset)
        else:
            raise Exception()

    def backprop(self, inputs, expectations):
        if type=="regressor":
            outputs=self(inputs)
            weighed_delta=np.array([(outputs-np.array(expectations))/len(outputs)])
            for i in range(1,len(self.layers)):
                weighed_delta=np.array(self.layers[-i].backprop(weighed_delta))
        elif self.__type=="classifier":
            outputs=self(inputs)
            weighed_delta=np.array([(outputs-np.array(expectations))/len(outputs)])*outputs(1-outputs)
            for i in range(1,len(self.layers)):
                weighed_delta=np.array(self.layers[-i].backprop(weighed_delta))
        else:
            raise Exception()

    def train(self, data):
        assert isinstance(data, dict)
        for key in data:
            self.backprop(np.array(key), np.array(data[key]))
        return 1



if __name__=="__main__":
    pass
