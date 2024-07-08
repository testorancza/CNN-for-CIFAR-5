from CONV2D import *
from Dense import *
from MaxPooling2D import *
from Optimizer import *
from ReLU import *
from Softmax import *
import pickle
import numpy as np

class CNN(object):

    def __init__(self, input_dimension=(3, 32, 32), number_of_filters=16, size_of_filter=3,
                 hidden_dimension=256, number_of_classes=5, weight_scale=1e-3, regularization=0.0,
                 dtype=np.float32):

        self.dtype = dtype
        self.params = {}
        self.regularization = regularization

        C, H, W = input_dimension
        HH = WW = size_of_filter
        F = number_of_filters
        Hh = hidden_dimension
        Hclass = number_of_classes

        self.params['w1'] = weight_scale * np.random.rand(F, C, HH, WW)
        self.params['b1'] = np.zeros(F)

        self.cnn_params = {'stride': 1, 'pad': int((size_of_filter - 1) / 2)}
        Hc = int(1 + (H + 2 * self.cnn_params['pad'] - HH) / self.cnn_params['stride'])
        Wc = int(1 + (W + 2 * self.cnn_params['pad'] - WW) / self.cnn_params['stride'])

        self.pooling_params = {'pooling_height': 2, 'pooling_width': 2, 'stride': 2}
        Hp = int(1 + (Hc - self.pooling_params['pooling_height']) / self.pooling_params['stride'])
        Wp = int(1 + (Wc - self.pooling_params['pooling_width']) / self.pooling_params['stride'])

        self.params['w2'] = weight_scale * np.random.rand(F * Hp * Wp, Hh)
        self.params['b2'] = np.zeros(Hh)

        self.params['w3'] = weight_scale * np.random.rand(Hh, Hclass)
        self.params['b3'] = np.zeros(Hclass)

        for key, value in self.params.items():
            self.params[key] = value.astype(dtype)

    def loss_function(self, x, y):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Forward pass:
        # Input --> Conv --> ReLU --> Pool --> FC --> ReLU --> FC --> Softmax
        conv2d_output, cache_conv2d = conv2d_forward(x, w1, b1, self.cnn_params)
        reLU_output_1, cache_reLU_1 = reLU_forward(conv2d_output)
        max_pooling_2d_output, cache_max_pooling_2d = max_pooling_2d_forward(reLU_output_1, self.pooling_params)
        dense_output, cache_dense = dense_forward(max_pooling_2d_output, w2, b2)
        reLU_output_2, cache_reLU_2 = reLU_forward(dense_output)
        scores, cache_dense_output = dense_forward(reLU_output_2, w3, b3)

        loss, d_scores = softmax_loss(scores, y)

        loss += 0.5 * self.regularization * np.sum(np.square(w1))
        loss += 0.5 * self.regularization * np.sum(np.square(w2))
        loss += 0.5 * self.regularization * np.sum(np.square(w3))

        dx3, dw3, db3 = dense_backward(d_scores, cache_dense_output)
        dw3 += self.regularization * w3

        d_reLU_2 = reLU_backward(dx3, cache_reLU_2)
        dx2, dw2, db2 = dense_backward(d_reLU_2, cache_dense)
        dw2 += self.regularization * w2

        d_max_pooling_2d = max_pooling_2d_backward(dx2, cache_max_pooling_2d)
        d_reLU_1 = reLU_backward(d_max_pooling_2d, cache_reLU_1)
        dx1, dw1, db1 = conv2d_backward(d_reLU_1, cache_conv2d)
        dw1 += self.regularization * w1

        gradients = dict()
        gradients['w1'] = dw1
        gradients['b1'] = db1
        gradients['w2'] = dw2
        gradients['b2'] = db2
        gradients['w3'] = dw3
        gradients['b3'] = db3

        return loss, gradients

    def forward_pass_to_softmax(self, x):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Forward pass:
        # Input --> Conv --> ReLU --> Pool --> FC --> ReLU --> FC --> Softmax
        conv2d_output, _ = conv2d_forward(x, w1, b1, self.cnn_params)
        reLU_output_1, _ = reLU_forward(conv2d_output)
        max_pooling_2d_output, _ = max_pooling_2d_forward(reLU_output_1, self.pooling_params)
        dense_output, _ = dense_forward(max_pooling_2d_output, w2, b2)
        reLU_output_2, _ = reLU_forward(dense_output)
        scores, _ = dense_forward(reLU_output_2, w3, b3)

        return scores

class Model(object):

    def __init__(self, model, data, **kwargs):
        self.model = model

        self.x_train = data['x_train']
        self.y_train = data['y_train']
        self.x_validation = data['x_validation']
        self.y_validation = data['y_validation']
        self.x_test = data['x_test']
        self.y_test = data['y_test']

        self.optimization_config = kwargs.pop('optimization_config', {})
        self.learning_rate_decay = kwargs.pop('learning_rate_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.number_of_epochs = kwargs.pop('number_of_epochs', 10)
        self.verbose_mode = kwargs.pop('verbose_mode', True)

        self.update_rule = adam
        self._reset()

    def _reset(self):
        self.current_epoch = 0
        self.best_validation_accuracy = 0
        self.best_params = {}
        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        self.optimization_configurations = {}
        for param in self.model.params:
            value = {k: v for k, v in self.optimization_config.items()}
            self.optimization_configurations[param] = value

    def _step(self):
        number_of_train_images = self.x_train.shape[0]

        batch_mask = np.random.choice(number_of_train_images, self.batch_size)
        x_batch = self.x_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, gradient = self.model.loss_function(x_batch, y_batch)

        for param, value in self.model.params.items():
            dw = gradient[param]
            config_for_current_param = self.optimization_configurations[param]
            next_w, next_configuration = self.update_rule(value, dw, config_for_current_param)
            self.model.params[param] = next_w
            self.optimization_configurations[param] = next_configuration

    def accuracy_check(self, x, y, number_of_samples=None, batch_size=100):
        N = x.shape[0]

        if number_of_samples is not None and N > number_of_samples:
            batch_mask = np.random.choice(N, number_of_samples)
            N = number_of_samples

            x = x[batch_mask]
            y = y[batch_mask]

        number_of_batches = int(N / batch_size)

        if N % batch_size != 0:
            number_of_batches += 1

        y_predicted = []

        for i in range(number_of_batches):
            start = i * batch_size
            end = (i + 1) * batch_size

            scores = self.model.forward_pass_to_softmax(x[start:end])
            y_predicted.append(np.argmax(scores, axis=1))

        y_predicted = np.hstack(y_predicted)
        accuracy = np.mean(y_predicted == y)

        return accuracy, y_predicted

    def fit(self):
        number_of_train_images = self.x_train.shape[0]
        iters_per_epoch = int(max(number_of_train_images / self.batch_size, 1))
        iterations_total = int(self.number_of_epochs * iters_per_epoch)

        for iter in range(iterations_total):
            self._step()

            end_of_current_epoch = (iter + 1) % iters_per_epoch == 0

            if end_of_current_epoch:
                self.current_epoch += 1
                for k in self.optimization_configurations:
                    self.optimization_configurations[k]['learning_rate'] *= self.learning_rate_decay

            first_iteration = (iter == 0)
            last_iteration = (iter == iterations_total - 1)

            if first_iteration or last_iteration or end_of_current_epoch:
                train_accuracy, _ = self.accuracy_check(self.x_train, self.y_train,number_of_samples=500)
                validation_accuracy, _ = self.accuracy_check(self.x_validation, self.y_validation)

                train_loss, _ = self.model.loss_function(self.x_train, self.y_train)
                validation_loss, _ = self.model.loss_function(self.x_validation, self.y_validation)

                self.train_accuracy_history.append(train_accuracy)
                self.validation_accuracy_history.append(validation_accuracy)
                self.train_loss_history.append(train_loss)
                self.validation_loss_history.append(validation_loss)

                if self.verbose_mode:
                    print('Epoch ' + str(self.current_epoch) + '/' + str(self.number_of_epochs) + ' | ' +
                          'loss: ' + str(round(train_loss,3)) + ' - ' + 'accuracy: ' + str(round(train_accuracy,3)) + ' - ' +
                          'val_loss: ' + str(round(validation_loss,3)) + ' - ' + 'val_accuracy: ' + str(round(validation_accuracy,3)))

                if validation_accuracy > self.best_validation_accuracy:
                    self.best_validation_accuracy = validation_accuracy
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v

        self.model.params = self.best_params

        history_dictionary = {'train_loss_history': self.train_loss_history,
                              'train_accuracy_history': self.train_accuracy_history,
                              'validation_loss_history': self.validation_loss_history,
                              'validation_accuracy_history': self.validation_accuracy_history
                              }

        with open('Model/model_history.pickle', 'wb') as f:
            pickle.dump(history_dictionary, f)

    def evaluate(self):
        test_accuracy, y_predicted = self.accuracy_check(self.x_test, self.y_test)
        test_loss, _ = self.model.loss_function(self.x_test, self.y_test)

        evaluation_results = {'y_pred': y_predicted, 'acc': test_accuracy, 'loss': test_loss}

        print('Test data evaluation | ' + 'accuracy: ' + str(round(test_accuracy,3)) + ' - '
              + 'loss: ' + str(round(test_loss,3)))

        with open('Model/evaluation_results.pickle', 'wb') as f:
            pickle.dump(evaluation_results, f)

    def save(self):
        with open('Model/model_params.pickle', 'wb') as f:
            pickle.dump(self.model.params, f)