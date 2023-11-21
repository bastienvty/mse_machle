import numpy as np
class MLP:
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        return 1.0 - a**2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        return a * (1 - a)

    def __init__(self, layers, activation='tanh'):
        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]
        self.layers = layers

        if activation == 'logistic':
            self.activation = self.__logistic
            self.activation_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.activation = self.__tanh
            self.activation_deriv = self.__tanh_deriv

        self.init_weights()

    def init_weights(self):
        self.weights = []
        for i in range(1, len(self.layers) - 1):
            self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i])) - 1) * 0.25)
        self.weights.append((2 * np.random.random((self.layers[i] + 1, self.layers[i + 1])) - 1) * 0.25)

    def fit(self, data_train, data_test=None, learning_rate=0.1, momentum=0.5, epochs=100):
        X = np.atleast_2d(data_train[0])
        y = np.array(data_train[1])
        error_train = np.zeros(epochs)

        if data_test is not None:
            error_test = np.zeros(epochs)

        a = []
        for l in self.layers:
            a.append(np.zeros(l))

        for k in range(epochs):
            error_it = np.zeros(X.shape[0])
            prev_delta_weights = [np.zeros_like(w) for w in self.weights]

            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])
                a[0] = X[i]

                for l in range(len(self.weights)):
                    a[l] = np.concatenate((a[l], np.ones(1)))
                    a[l+1] = self.activation(np.dot(a[l], self.weights[l]))

                error = a[-1] - y[i]
                error_it[it] = np.mean(error ** 2)
                deltas = [error * self.activation_deriv(a[-1])]

                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                    deltas[-1] = deltas[-1][:-1]

                deltas.reverse()

                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])

                    delta_weights = -learning_rate * layer.T.dot(delta) + momentum * prev_delta_weights[i]
                    self.weights[i] += delta_weights

                    prev_delta_weights[i] = delta_weights

            error_train[k] = np.mean(error_it)

            if data_test is not None:
                error_test[k], _ = self.compute_MSE(data_test)

        if data_test is None:
            return error_train
        else:
            return error_train, error_test

    def predict(self, x):
        a = np.array(x)
        for l in range(0, len(self.weights)):
            temp = np.ones(a.shape[0]+1)
            temp[0:-1] = a
            a = self.activation(np.dot(temp, self.weights[l]))
        return a

    def compute_output(self, data):
        assert len(data.shape) == 2, 'data must be a 2-dimensional array'

        out = np.zeros((data.shape[0], self.n_outputs))
        for r in np.arange(data.shape[0]):
            out[r, :] = self.predict(data[r, :])
        return out

    def compute_MSE(self, data_test):
        assert len(data_test[0].shape) == 2, 'data[0] must be a 2-dimensional array'

        out = self.compute_output(data_test[0])
        return np.mean((data_test[1] - out) ** 2), out



class MLP_N_output_classes:
    def __tanh(self, x):
        return np.tanh(x)

    def __tanh_deriv(self, a):
        return 1.0 - a**2

    def __logistic(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def __logistic_derivative(self, a):
        return a * (1 - a)

    def __softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def __init__(self, layers, activation='tanh'):
        self.n_inputs = layers[0]
        self.n_outputs = layers[-1]
        self.layers = layers

        if activation == 'logistic':
            self.activation = self.__logistic
            self.activation_deriv = self.__logistic_derivative
        elif activation == 'tanh':
            self.activation = self.__tanh
            self.activation_deriv = self.__tanh_deriv
        elif activation == 'softmax':
            self.activation = self.__softmax
            self.activation_deriv = lambda x: x  # Derivative of softmax is the identity function

        self.init_weights()

    def init_weights(self):
        self.weights = []
        for i in range(1, len(self.layers) - 1):
            self.weights.append((2 * np.random.random((self.layers[i - 1] + 1, self.layers[i])) - 1) * 0.25)
        # For the output layer
        self.weights.append((2 * np.random.random((self.layers[-2] + 1, self.layers[-1])) - 1) * 0.25)

    def fit(self, data_train, data_test=None, learning_rate=0.1, momentum=0.5, epochs=100):
        X = np.atleast_2d(data_train[0])
        y = np.array(data_train[1])
        error_train = np.zeros(epochs)

        if data_test is not None:
            error_test = np.zeros(epochs)

        a = []
        for l in self.layers:
            a.append(np.zeros(l))

        for k in range(epochs):
            error_it = np.zeros(X.shape[0])
            prev_delta_weights = [np.zeros_like(w) for w in self.weights]

            for it in range(X.shape[0]):
                i = np.random.randint(X.shape[0])
                a[0] = X[i]

                for l in range(len(self.weights)):
                    a[l] = np.concatenate((a[l], np.ones(1)))
                    a[l+1] = self.activation(np.dot(a[l], self.weights[l]))

                error = a[-1] - y[i]
                error_it[it] = np.mean(error ** 2)
                deltas = [error * self.activation_deriv(a[-1])]

                for l in range(len(a) - 2, 0, -1):
                    deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
                    deltas[-1] = deltas[-1][:-1]

                deltas.reverse()

                for i in range(len(self.weights) - 1):  # Exclude the output layer
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])

                    delta_weights = -learning_rate * layer.T.dot(delta) + momentum * prev_delta_weights[i]
                    self.weights[i] += delta_weights

                    prev_delta_weights[i] = delta_weights

            error_train[k] = np.mean(error_it)

            if data_test is not None:
                error_test[k], _ = self.compute_MSE(data_test)

        if data_test is None:
            return error_train
        else:
            return error_train, error_test

    def predict(self, x):
        a = np.array(x)
        for l in range(0, len(self.weights) - 1):  # Exclude the output layer
            temp = np.ones(a.shape[0] + 1)
            temp[0:-1] = a
            a = self.activation(np.dot(temp, self.weights[l]))

        # Apply softmax for the output layer
        temp = np.ones(a.shape[0] + 1)
        temp[0:-1] = a
        a = self.__softmax(np.dot(temp, self.weights[-1]))

        return a

    def compute_output(self, data):
        assert len(data.shape) == 2, 'data must be a 2-dimensional array'

        out = np.zeros((data.shape[0], self.n_outputs))
        for r in np.arange(data.shape[0]):
            out[r, :] = self.predict(data[r, :])
        return out

    def compute_MSE(self, data_test):
        assert len(data_test[0].shape) == 2, 'data[0] must be a 2-dimensional array'

        out = self.compute_output(data_test[0])
        return np.mean((data_test[1] - out) ** 2), out
