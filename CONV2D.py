import numpy as np

def conv2d_forward(x, w, b, cnn_params):

    stride = cnn_params['stride']
    pad = cnn_params['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    cache = (x, w, b, cnn_params)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    height_out = int(1 + (H + 2 * pad - HH) / stride)
    width_out = int(1 + (W + 2 * pad - WW) / stride)

    feature_maps = np.zeros((N, F, height_out, width_out))

    for n in range(N):
        for f in range(F):
            height_index = 0
            for i in range(0, H, stride):
                width_index = 0
                for j in range(0, W, stride):
                    feature_maps[n, f, height_index, width_index] = \
                        np.sum(x_padded[n, :, i:i+HH, j:j+WW] * w[f, :, :, :]) + b[f]
                    width_index += 1
                height_index += 1

    return feature_maps, cache

def conv2d_backward(derivative_out, cache):
    x, w, b, cnn_params = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, height_out, weight_out = derivative_out.shape

    stride = cnn_params['stride']
    pad = cnn_params['pad']

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for n in range(N):
        for f in range(F):
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    dx_padded[n, :, i:i+HH, j:j+WW] += w[f, :, :, :] * derivative_out[n, f, i, j]
                    dw[f, :, :, :] += x_padded[n, :, i:i+HH, j:j+WW] * derivative_out[n, f, i, j]
                    db[f] += derivative_out[n, f, i, j]

    dx = dx_padded[:, :, 1:-1, 1:-1]

    return dx, dw, db