import numpy as num
import utils



class Neuron(object):

    _weights = []
    _input = []
    _bias = None

    def __init__(self,weights = [], bias = -1):
        self._weights = weights
        self._bias = bias

    def run(self, input):
        input = [self._bias] + input
        sum = num.dot(input,self._weights)
        return utils.escada(sum)

    @property
    def w(self):
        return self._weights

    @w.setter
    def weigths(self, weigths):
        self._weigths = weigths



