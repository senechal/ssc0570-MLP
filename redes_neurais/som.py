import numpy


class SOM:
    def __init__(self,x, y, input_size, sigma=1.0, n=0.1, random_seed=None):
        self._random = numpy.random.RandomState(random_seed)
        self._n = n
        self._sigma = sigma
        self._w = self._random.rand(x,y,input_size)
        for i in range(x):
            for j in range(y):
                self._w[i,j] = self._w[i,j]/ numpy.linalg.norm(self._w[i,j])
        self._map = numpy.zeros((x,y))
        self._x = x
        self._y = y
        self._input_size = input_size

    @property
    def weights(self):
        return self._w

    def train_random(self, data, max_t):
        self._max_t = max_t
        for t in range(max_t):
            index = self._random.randint(len(data))
            random_input = data[index]
            winner_index = self._winner(random_input[:self._input_size])
            self._update(random_input[:self._input_size], winner_index, t)

    def train_batch(self, data):
        self._max_t = len(data)
        for index, x_input in enumerate(data):
            winner_index = self._winner(x_input[:self._input_size])
            self._update(x_input[:self._input_size], winner_index, index)

    def random_weights_init(self, data):
        iterator = numpy.nditer(self._map, flags=['multi_index'])
        while not iterator.finished:
            self._w[iterator.multi_index] = data[self._random.randint(len(data))][:-1]
            self._w[iterator.multi_index] = self._w[iterator.multi_index]/numpy.linalg.norm(self._w[iterator.multi_index])
            iterator.iternext()

    def decay_function(self, x, t, T):
        return x/(1+t/(T/2))

    def winner(self, x):
        return self._winner(x[:self._input_size])

    def _winner(self,x):
        sub = numpy.subtract(x, self._w)
        iterator = numpy.nditer(self._map, ['multi_index'])
        while not iterator.finished:
            self._map[iterator.multi_index] = numpy.linalg.norm(sub[iterator.multi_index])
            iterator.iternext()
        return numpy.unravel_index(self._map.argmin(), self._map.shape)

    def _gaussian(self, center, sigma):
        d = 2*numpy.pi*sigma*sigma
        ax = numpy.exp(-numpy.power(numpy.arange(self._x)-center[0], 2)/d)
        ay = numpy.exp(-numpy.power(numpy.arange(self._y)-center[1], 2)/d)
        return numpy.outer(ax,ay)

    def _update(self, x, winner_index, t):
        n = self.decay_function(self._n , t, self._max_t)
        sigma = self.decay_function(self._sigma, t, self._max_t)
        delta = self._gaussian(winner_index, sigma) * n
        iterator = numpy.nditer(delta, flags=['multi_index'])
        while not iterator.finished:
            self._w[iterator.multi_index] += delta[iterator.multi_index]*(x-self._w[iterator.multi_index])
            self._w[iterator.multi_index] = self._w[iterator.multi_index]/numpy.linalg.norm(self._w[iterator.multi_index])
            iterator.iternext()

    def distance_map(self):
        um = zeros((self.weights.shape[0], self.weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += numpy.linalg.norm(self.weights[ii, jj, :]-self.weights[it.multi_index])
            it.iternext()
        um = um/um.max()
        return um
