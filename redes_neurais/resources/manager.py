import json
import logging
from redes_neurais.mlp import Mlp
from PIL import Image
import numpy
import pprint

class DataProcessor(object):
    def __init__(self, path_to_train, path_to_test):
        self.path_to_train = path_to_train
        self.path_to_test = path_to_test

    @classmethod
    def openJson(cls, path_to_json):
        try:
            with open(path_to_json) as data_file:
                return json.load(data_file)
        except Exception as e:
            print e

    def run(self):
        train_data = self.openJson(self.path_to_train)
        test_data = self.openJson(self.path_to_test)
        try:
            data_format_train = train_data['format']
            data_format_test = test_data['format']
            if data_format_train != data_format_test:
                raise Exception("formats don't match.")
            else:
                data_format = data_format_train
        except Exception as e:
            print e
            return None

        return self._dataFormatProcessor(data_format, train_data, test_data)

    def _open_bmp(self, path_to_bmp):
        img = Image.open(path_to_bmp)
        return numpy.matrix(img).getA1().tolist()


    def _dataFormatProcessor(self, data_format, train_data, test_data):
        if data_format == "binary":
            return (train_data, test_data)
        elif data_format == "bmp":
            for index, item in enumerate(train_data["data"]):
                train_data["data"][index] = self._open_bmp(item)
            train_data["input_size"] = len(train_data["data"][0])
            for index, item in enumerate(test_data["data"]):
                test_data["data"][index] = self._open_bmp(item)
            return (train_data, test_data)
        else:
            raise NotImplementedError("Format not implemented.")



def classify(class_values, results):
    pp = pprint.PrettyPrinter(indent=4)
    classes = dict()
    pp.pprint(results)
    for index, item in enumerate(class_values):
        classes.update({"class_{}".format(index):{"format":item, "member":[]}})

    for index, result in enumerate(results):
        for key, _class in classes.items():
            if numpy.allclose(numpy.array(result), numpy.array(_class["format"]),rtol=0.01, atol=0.01):
                _class["member"].append(result.pop(index))
    classes.update({"Unclassified":results})
    pp.pprint(classes)

def run_mlp(path_to_config, path_to_train, path_to_test):
    processor = DataProcessor(path_to_train, path_to_test)
    config = DataProcessor.openJson(path_to_config)
    values = processor.run()
    if values:
        train_data, test_data = values
        mlp = Mlp(train_data["input_size"], config["hidden"], config["output"])
        mlp.train(train_data["data"], train_data["targets"], config["max_iterations"], config["min_error"], config["n"], config["m"])
        results, iterations = mlp.test(test_data["data"])
        print "Iteration: {}".format(iterations)
        classify(train_data["targets"], results)
