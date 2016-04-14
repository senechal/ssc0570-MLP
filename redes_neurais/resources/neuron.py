import numpy
import utils



class Neuron(object):

    _weights = numpy.array([])
    _input = numpy.array([])

    def __init__(self,weights):
        self._weights = numpy.array(weights)

    def run(self, input):
        sum = numpy.dot(input,self._weights)
        return sum

    @property
    def w(self):
        return self._weights
    @w.setter
    def w(self, weigths):
        self._weights = numpy.array(weigths)

    @classmethod
    def step(cls,v):
        return 1 if v > 0 else 0

    @classmethod
    def step_tetha(cls,v, t):
        return 1 if v>t else 0

    @classmethod
    def sigmoid(cls,v,a):
        return 1/(1+numpy.exp(v*a))
