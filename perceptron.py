import numpy as np
from math import *
import random


class Perceptron:

    def __init__(self, inputs, l_rate=0.05, function="logistic"):
        assert type(inputs)==int
        self.weights=2*np.random.rand(inputs+1)-1
        if function=="relu":
            self.weights=self.weights/4
        self.l_rate=l_rate
        self.function=function

    def set_attr(self, attrname, value):
        if attrname=="function":
            self.function=value
        elif attrname=="l_rate":
            self.l_rate=l_rate
        else:
            raise Exception("Wrong key")


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


if __name__=="__main__":
    pass
