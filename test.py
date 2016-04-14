from redes_neurais.perceptron.perceptron import Perceptron
from redes_neurais.resources.neuron import Neuron
x = [[-1, 0, 0], [-1, 0, 1], [-1, 1, 0], [-1, 1, 1]]
e = [0, 1, 1, 1]
p = Perceptron(e)
p.x = x
ni = 0.1
w = [0, 0, 0]
n = Neuron(w)
a = p.train_neuron(n, ni)
print a.w
