import numpy as np
from math import *
from layer import *


class Network:
    def __init__(self, inputs, l_rate=0.1, ntype="regressor", func="logistic"):
        assert type(inputs)==int
        self.inputs=inputs
        self.layers=[InputLayer(inputs)]
        self.l_rate=l_rate
        self.__type=ntype
        self.default_function=func

    def add_layer(self, nodes, function=None):
        if function==None:
            function=self.default_function
        self.layers.append(Layer(self.layers[len(self.layers)-1], nodes, l_rate=self.l_rate, function=function))
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
                if np.where(np.array(outputs)==max(outputs))[0]!=np.where(np.array(dataset[key])==max(dataset[key]))[0]:
                    mistakes+=1
            return mistakes/len(dataset)
        else:
            raise Exception()

    def backprop(self, inputs, expectations):
        if self.__type=="regressor":
            outputs=np.array(self(inputs))
            weighed_delta=np.array([(outputs-np.array(expectations))/len(outputs)])
            for i in range(1,len(self.layers)):
                weighed_delta=np.array(self.layers[-i].backprop(weighed_delta))
        elif self.__type=="classifier":
            outputs=np.array(self(inputs))
            weighed_delta=np.array([(outputs-np.array(expectations))/len(outputs)])*outputs*(1-outputs)
            for i in range(1,len(self.layers)):
                weighed_delta=np.array(self.layers[-i].backprop(weighed_delta))
        else:
            raise Exception()

    def train(self, data):
        assert isinstance(data, dict)
        for key in random.sample(list(data), len(data)):
            self.backprop(np.array(key), np.array(data[key]))
        return 1



if __name__=="__main__":
    pass
