import numpy as num

def sigmoid(x):
    return num.tanh(x)

def escada(x):
    return 1 if x > 0 else 0

def wi1(w0, ni, y, e, xi):
    w1 = []
    for index, value in enumerate(w0):
        w1[index] = value + ni*(y - e)*xi[index]
    return w1

