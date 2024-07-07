import numpy as np
def max_pooling_2d_forward(x, pooling_params):
    N, F, H, W = x.shape

    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']

    cache = (x, pooling_params)

    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)
    pooled_output = np.zeros((N, F, height_pooled_out, width_polled_out))

    for n in range(N):
        for i in range(height_pooled_out):
            for j in range(width_polled_out):
                ii = i * stride
                jj = j * stride
                current_pooling_region = x[n, :, ii:ii+pooling_height, jj:jj+pooling_width]
                pooled_output[n, :, i, j] = \
                    np.max(current_pooling_region.reshape((F, pooling_height * pooling_width)), axis=1)

    return pooled_output, cache

def max_pooling_2d_backward(derivatives_out, cache):
    x, pooling_params = cache
    N, F, H, W = x.shape

    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']

    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)

    dx = np.zeros((N, F, H, W))

    for n in range(N):
        for f in range(F):
            for i in range(height_pooled_out):
                for j in range(width_polled_out):
                    ii = i * stride
                    jj = j * stride
                    current_pooling_region = x[n, f, ii:ii+pooling_height, jj:jj+pooling_width]
                    current_maximum = np.max(current_pooling_region)
                    temp = current_pooling_region == current_maximum
                    dx[n, f, ii:ii+pooling_height, jj:jj+pooling_width] += \
                        derivatives_out[n, f, i, j] * temp

    return dx