import numpy as np

def softmax_loss(x, y):

    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probabilities = shifted_logits - np.log(z)
    probabilities = np.exp(log_probabilities)

    N = x.shape[0]

    loss = -np.sum(log_probabilities[np.arange(N), y]) / N

    dx = probabilities
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx