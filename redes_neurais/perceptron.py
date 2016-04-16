from redes_neurais.resources.neuron import Neuron
import numpy


class Perceptron(object):

    _oracle = numpy.array([])
    _input = numpy.matrix([])
    _trained_w = numpy.array([])
    _step_w = []

    def __init__(self, oracle, label="", bias=-1):
        self._oracle = numpy.array(oracle)
        self._label = label
        self.bias = bias

    @property
    def x(self):
        return self._input

    @x.setter
    def x(self, input):
        self._input = numpy.matrix(input)

    def _recalc_w(self, w0, ni, d, y, e):
        return w0 + ni*(d-y)*e

    def train_neuron(self, n, ni=0.1):
        w = n.w
        for index, row in enumerate(self._input.A):
            bias_row = numpy.append([self.bias], row)
            v = n.run(bias_row.tolist())
            y = Neuron.step(v)
            n.w = self._recalc_w(n.w, ni, self._oracle[index], y, bias_row)
            self._step_w.append(n.w)
        if numpy.array_equal(n.w, w):
            return n
        else:
            return self.train_neuron(n, ni)

    def test(self, n, tests):
        data = numpy.matrix(tests)
        print "Number of steps (n): {}".format(len(self._step_w))
        print "Trained Ws {}".format(n.w)
        for row in data.A:
            bias_row = numpy.append([self.bias], row)
            v = n.run(bias_row.tolist())
            y = Neuron.step(v)
            print "Testing {0}: output:{1}".format(row, y)
