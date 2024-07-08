import numpy as np

def reLU_forward(x):
    cache = x
    reLU_output = np.maximum(0, x)

    return reLU_output, cache

def reLU_backward(derivatives_out, cache):
    x = cache
    temp = x > 0
    dx = temp * derivatives_out

    return dx