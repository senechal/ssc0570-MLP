import json
import logging
from redes_neurais.perceptron import Perceptron
from redes_neurais.resources.neuron import Neuron

logger = logging.getLogger('neural_network')


class DataProcessor(object):
    def __init__(self, path_to_train, path_to_test):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test

    def _openJson(self, path_to_json):
        try:
            with open(path_to_json) as data_file:
                return json.load(data_file)
        except Exception as e:
            logger.info(e)

    def run(self):
        train_data = self._openJson(self.path_to_train)
        test_data = self._openJson(self.path_to_test)
        try:
            data_format_train = train_data['format']
            data_format_test = test_data['format']
            if data_format_train != data_format_test:
                raise Exception("formats don't match.")
            else:
                data_format = data_format_train
        except Exception as e:
            logger.info(e)
            return None

        return self._dataFormatProcessor(data_format, train_data, test_data)

    def _dataFormatProcessor(self, data_format, train_data, test_data):
        if data_format == "binary":
            return (train_data, test_data)
#        elif data_format == "bmp":
#            pass
        else:
            raise NotImplementedError("Format not implemented.")


def run_perceptron(path_to_train, path_to_test):
    processor = DataProcessor(path_to_train, path_to_test)
    values = processor.run()
    if values:
        train_data, test_data = values
        perceptron = Perceptron(train_data['oracle'])
        perceptron.x = train_data['data']
        neuron = Neuron(train_data['weights'])
        trained_neuron = perceptron.train_neuron(neuron, train_data['n'])
        perceptron.test(trained_neuron, test_data['data'])
