from redes_neurais.resources.neuron import Neuron
import numpy

class Perceptron(object):

    _oracle = numpy.array([])
    _input = numpy.matrix([])
    _trained_w = numpy.array([])

    def __init__(self, oracle, label = ""):
        self._oracle = numpy.array(oracle)
        self._label = label

    @property
    def x(self):
        return self._input
    @x.setter
    def x(self, input):
        self._input = numpy.matrix(input)

    def _recalc_w(self,w0, ni,d,y,e):
        return w0 + ni*(d-y)*e


    def train_neuron(self, n, ni=0.1):
        w = n.w
        for index, row in enumerate(self._input.A):
            v = n.run(row.tolist())
            y = Neuron.step(v)
            n.w = self._recalc_w(n.w, ni, self._oracle[index], y, row)
        if numpy.array_equal(n.w, w):
            return n
        else:
            return self.train_neuron(n, ni)


