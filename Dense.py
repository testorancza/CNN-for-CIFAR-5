import numpy as np

def dense_forward(x, w, b):

    cache = (x, w, b)

    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    dense_output = np.dot(x_reshaped, w) + b

    return dense_output, cache

def dense_backward(derivatives_out, cache):
    x, w, b = cache

    dx = np.dot(derivatives_out, w.T).reshape(x.shape)
    N = x.shape[0]
    x = x.reshape(N, -1)
    dw = np.dot(x.T, derivatives_out)
    db = np.dot(np.ones(dx.shape[0]), derivatives_out)

    return dx, dw, db