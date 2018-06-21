"""A simple fully-connected neural-network, using gradient descent"""

import numpy as np
from math import *
from layer import *


class Network:
    """ Initiate the neural network. Requires adding layers to operate.

        inputs    number of inputs: [int]\n
        l_rate    learning rate: [float] between 0 and 1 not inclusive\n
        ntype     network type: [str] "regressor" or "classifier"\n
        function  default activation function: [str] "logistic", "relu" (not working), or None"""


    def __init__(self, inputs, l_rate=0.1, ntype="regressor", function="logistic"):
        assert type(inputs)==int
        self.inputs=inputs
        self.layers=[InputLayer(inputs)]
        assert 0<l_rate and 1>l_rate
        self.l_rate=l_rate
        self.__type=ntype
        self.default_function=function

    def add_layer(self, nodes, function=None):
        """Add a layer to the network. \n

           nodes     number of neurons: [int]
           function  activation function: [str] "logistic", "relu" (not working), or None"""


        if function==None:
            function=self.default_function
        self.layers.append(Layer(self.layers[len(self.layers)-1], nodes, l_rate=self.l_rate, function=function))
        return self

    def __call__(self, inputs):
        """Forward propagation. \n

            inputs    network inputs: np.ndarray of [float]"""
        inputs=np.array(inputs)
        outputs=inputs
        for layer in self.layers:
            outputs=layer(outputs)

        if self.__type=="regressor":
            return outputs
        elif self.__type=="classifier":
            exp_sum=sum([exp(output) for output in outputs])
            outputs=[exp(output)/exp_sum for output in outputs]
            return outputs
        else:
            raise Exception()

    def evaluate(self, dataset):
        """Evaluate the network.

            dataset   dictionary of inputs to outputs"""


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
        """Perform backpropagation.

            inputs        network inputs: ndarray of [float]
            expectations  expected output values: ndarray of [float] for regressor, ndarray of [str] for classifier"""


        outputs=np.array(self(inputs))

        if self.__type=="regressor":
            weighed_delta=np.array([(outputs-np.array(expectations))/len(outputs)])
        elif self.__type=="classifier": #because of additional function
            weighed_delta=np.array([(outputs-np.array(expectations))/len(outputs)])*outputs*(1-outputs)
        else:
            raise Exception()

        for i in range(1,len(self.layers)):
            weighed_delta=np.array(self.layers[-i].backprop(weighed_delta))

    def train(self, dataset):
        """Train the network with iterative backpropagation.

            dataset   dictionary of inputs to outputs"""


        assert isinstance(dataset, dict)
        for key in random.sample(list(dataset), len(dataset)):
            self.backprop(np.array(key), np.array(dataset[key]))
        return 1



if __name__=="__main__":
    pass
