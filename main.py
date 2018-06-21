from network import *
import pandas as pd
import random
from datetime import datetime

or_train_set={(1,1):1,
           (1,0):1,
           (0,1):1,
           (0,0):0}

xor_train_set={(1,1):0,
           (1,0):1,
           (0,1):1,
           (0,0):0}

and_train_set={(1,1):1,
           (1,0):0,
           (0,1):0,
           (0,0):0}

xand_train_set={(1,1):1,
           (1,0):0,
           (0,1):0,
           (0,0):1}

train_set={}
for key in or_train_set:
    train_set[(*key,1,0,0,0)]=or_train_set[key]
for key in xor_train_set:
    train_set[(*key,0,1,0,0)]=xor_train_set[key]
for key in and_train_set:
    train_set[(*key,0,0,1,0)]=and_train_set[key]
for key in xand_train_set:
    train_set[(*key,0,0,0,1)]=xand_train_set[key]

print(train_set)


nn=Network(6, 0.3)
nn.add_layer(8)
nn.add_layer(8)
nn.add_layer(8)
nn.add_layer(1)
start=datetime.now()
for i in range(4000):

    if i%100==0:
        print("generation", i, ":", (datetime.now()-start).seconds+(datetime.now()-start).microseconds/10**6,"seconds elapsed")
        print("error", nn.evaluate(train_set))
    nn.train(train_set)

"""for key in random.sample(list(xor_train_set), 10):
    print(nn(key), train_set[key])
"""
for key in train_set:
    print(nn(key), train_set[key])
    #print(nn.layers[2].neurons[0].weights)
