import numpy
from resources.utils import random_list, sigmoid, d_sigmoid
import pprint


class Mlp:

    _iterations = 0
    _input = []
    _hidden_neurons = []
    _output_neurons = []

    def __init__(self, input_size, hidden_neurons_size, output_neurons_size):
        self._x_size = input_size + 1
        self._hn_size = hidden_neurons_size
        self._on_size = output_neurons_size

        self._input = [0]*self._x_size
        for i in range(self._hn_size):
            self._hidden_neurons.append({
                            "y":0,
                            "w":random_list(self._x_size, -0.2, 0.2),
                            "last_delta": [0]*self._x_size})
        for i in range(self._on_size):
            self._output_neurons.append({
                            "y":0,
                            "w":random_list(self._hn_size, -0.2, 0.2),
                            "last_delta": [0]*self._hn_size})

    def propagate(self, input):
        if len(input) == self._x_size-1:
            self._input = input + [1]
        yh = []
        for index, neuron in enumerate(self._hidden_neurons):
            v = numpy.dot(self._input, neuron['w'])
            self._hidden_neurons[index]["y"] = sigmoid(v)
            yh.append(self._hidden_neurons[index]["y"])

        yo = []
        for index, neuron in enumerate(self._output_neurons):
            v = numpy.dot(yh, neuron["w"])
            self._output_neurons[index]["y"] = sigmoid(v)
            yo.append(self._output_neurons[index]["y"])
        return yo



    def backPropagate (self, target, N, M):
        o_deltas = []
        for index, neuron in enumerate(self._output_neurons):
            e = target[index] - neuron["y"]
            o_deltas.append(e* d_sigmoid(neuron["y"]))

        for i, hidden_neuron in enumerate(self._hidden_neurons):
            for j, output_neuron in enumerate(self._output_neurons):
                aux = o_deltas[j] * hidden_neuron["y"]
                self._output_neurons[j]["w"][i] += N*aux +M*output_neuron["last_delta"][i]
                self._output_neurons[j]["last_delta"][i] = aux

        h_deltas = []
        for i, hidden_neuron in enumerate(self._hidden_neurons):
            e = 0
            for j, output_neuron in enumerate(self._output_neurons):
                e += o_deltas[j] * output_neuron["w"][i]
            h_deltas.append(e*d_sigmoid(hidden_neuron["y"]))

        for i, input in enumerate(self._input):
            for j, neuron in enumerate(self._hidden_neurons):
                aux = h_deltas[j] * input
                self._hidden_neurons[j]["w"][i] += N*aux + M*neuron["last_delta"][i]
                self._hidden_neurons[j]["last_delta"][i] = aux

        error = 0
        for i in range(len(target)):
            error += 0.5 * numpy.power(target[i]-self._output_neurons[i]["y"], 2)
        return error

    def show_network(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self._hidden_neurons)
        pp.pprint(self._output_neurons)

    def test(self, inputs):
        tests = []
        for item in inputs:
            tests.append(self.propagate(item))
        return tests, self._iterations

    def train (self, inputs, targets, max_iterations = 1000, min_error=0.00001 ,N=0.5, M=0.1):
        for i in range(max_iterations):
            for index, item in enumerate(inputs):
                _input = item
                _target = targets[index]
                self.propagate(_input)
                error = self.backPropagate(_target, N, M)
            if i % 50 == 0:
                print 'Combined error:', error, 'iteration:', self._iterations
            self._iterations +=1
            if error < min_error:
                break;
