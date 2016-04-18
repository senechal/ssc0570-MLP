import numpy
import random

def random_list(size, init, end):
    _list = [0]*size
    for i in range(size):
        _list[i] = random.uniform(init, end)
    return _list

def sigmoid(v):
    return numpy.tanh(v)

def d_sigmoid(y):
    return 1 - numpy.power(y, 2)
