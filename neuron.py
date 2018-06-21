import numpy as np
from math import *
from perceptron import *


class Neuron(Perceptron):

    def __init__(self, input_layer, l_rate=0.1, function="logistic", weight_penalty=0.0001):
        super().__init__(len(input_layer), l_rate, function)
        self.input_layer=input_layer
        self.output=None
        self.weight_penalty=weight_penalty

    def __call__(self, inputs):
        output=super().__call__(inputs)
        self.output=output
        return output

    def deriv(self, x=None):
        if x==None:
            if self.function=="logistic":
                return self.output*(1-self.output)
            elif self.function=="relu":
                if self.output<0:
                    return 0
                else:
                    return 1
            elif self.function==None:
                return 1
            else:
                raise Exception()
        else:
            if self.function=="logistic":
                return exp(x)/(1+exp(x))**2
            elif self.fuction=="relu":
                if x<0:
                    return 0
                else:
                    return 1
            elif self.function==None:
                return 1
            else:
                raise Exception()

    def train(self, weighed_deltas):
        my_delta=sum(self.deriv()*weighed_deltas)
        prev_outputs=self.input_layer.output_cache
        self.weights-=self.l_rate*np.concatenate((prev_outputs, np.array([1])), axis=0)*my_delta+np.array(list(map(lambda x: abs(x)/x*self.weight_penalty, self.weights)))
        return my_delta*self.weights

if __name__=="__main__":
    pass
