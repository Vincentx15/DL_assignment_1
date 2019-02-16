import scipy as np
from scipy.special import softmax
import time
import random
import matplotlib.pyplot as plt
import copy


class NN:

    def __init__(self, hidden_dims=(700, 300), input_size=784, output_size=10, init_method=2,
                 non_linearity='relu', batch_size=16, alpha=0.01, lambd=0.1, save_path=None):
        """
        Initializes the NN with all its parameters
        :param hidden_dims:
        :param input_size:
        :param output_size:
        :param init_method:
        :param non_linearity:
        :param batch_size:
        :param alpha:
        :param lambd:
        :param save_path:
        """

        # Load a trained network
        if save_path is not None:
            self.load(save_path)
            self.grads = list(range(self.n_grad))
            return

        # Else, initializes a vanilla network
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.non_linearity = non_linearity
        self.layers = self.initialise_weights()
        self.cache = []
        self.batch_size = batch_size
        self.alpha = alpha
        self.lambd = lambd
        self.n_grad = len(self.layers)
        self.grads = list(range(self.n_grad))

    def initialise_weights(self):
        """
        Initializes the weights of the layers in the NN
        :return: initialized layers
        """

        def create_shape(shape, method=None):
            """
            Auxiliary function to build the layer matrices
            :param shape: shape of the matrix to build
            :param method: to create both gradients and matrices, defaults to init_methods
            :return:
            """
            # For gradients, just put zeroes
            if method is None:
                method = self.init_method

            # Then initialise
            if method == 0:
                return np.zeros(shape=shape)
            elif method == 1:
                tmp = np.randn(*shape) * 0.1
                tmp[:, -1] = 0
                return tmp
            else:
                d = np.sqrt(6.0 / np.sum(shape))
                tmp = np.random.uniform(low=-d, high=d, size=shape)
                tmp[:, -1] = 0
                return tmp

        dims = (self.input_size, *self.hidden_dims, self.output_size)
        layers = []
        for i in range(len(dims) - 1):
            shape = (dims[i + 1], dims[i] + 1)
            layer = create_shape(shape)
            layers.append(layer)
        return layers

    def activation(self, input, method):
        """
        Perform the activation of an input
        :param input: input of the activation
        :param method: method of activation
        :return: activated input
        """
        if method == 'relu':
            input = np.copy(input)
            input[input < 0] = 0
            return input
        if method == 'sigmoid':
            return 1 / (1 + np.exp(-np.asarray(input)))
        if method == 'tanh':
            return np.tanh(input)
        else:
            raise ValueError('Wrong method for activation')

    def activation_derivative(self, input, method):
        """
        Return the derivative of the activation at the input
        :param input: input of the activation
        :param method: method of activation
        :return: derivative of activation
        """
        if method == 'relu':
            return np.sign(input) / 2 + 0.5
        if method == 'sigmoid':
            s = 1 / (1 + np.exp(-np.asarray(input)))
            return s * (1 - s)
        if method == 'tanh':
            return 1 - np.tanh(input) ** 2
        else:
            raise ValueError('Wrong method for activation')

    def forward(self, input):
        """
        Propagate a forward pass
        :param input: should be padded, corresponds to h
        :return: preactivation a
        """
        self.cache = []  # Re-initiliazes the cache (case we make a forward pass, filling it, eg when evaluating)

        for i, layer in enumerate(self.layers):
            out = np.dot(layer, input)

            if np.isnan(out).any():
                print(i)
                raise ValueError('a', i)

            # The cache contains the preactivations except the input and the last one (that is fed
            # to a softmax so the computation is a bit different
            self.cache.append(out)
            input = np.concatenate((self.activation(out, self.non_linearity), np.ones((1, out.shape[1]))), axis=0)

            if np.isnan(input).any():
                raise ValueError('input', i)

            # Check if shape is not such as (25,), should not happen when using batch, but could happen with single pass
            # or batch of size one
            try:
                input.shape[1]
            except IndexError:
                input = input[:, np.newaxis]

        # remove last activation from cache (it gets a computation with the loss directly) and compute the final output
        self.cache.pop()
        final = softmax(out, axis=0)

        if np.isnan(final).any():
            print('out', out)
            print(np.exp(out))
            raise ValueError('final', i)

        return final

    def backwards(self, input, output, labels, lambd=None):
        """
        :param input: not padded
        :param output: a list outputed by softmax
        :param labels: as a float
        :param lambd: lambda parameter
        :return:
        """
        if lambd is None:
            lambd = self.lambd

        # vector of -out(i) except for the label where it is 1-out(i)
        delta = output
        try:
            n = len(labels)
            for i in range(n):
                delta[labels[i], i] -= 1
        except TypeError:
            delta[labels] += 1

        for i, a in enumerate(reversed(self.cache)):
            # compute activation again and also d(h(a))/d(a) (useful for propagation of delta)
            h = np.concatenate((self.activation(a, self.non_linearity), np.ones((1, a.shape[1]))), axis=0)

            dh = self.activation_derivative(a, self.non_linearity)

            # sign sometimes don't add dimension sometimes yes
            try:
                dh.shape[1]
            except IndexError:
                dh = dh[:, np.newaxis]

            # compute grads
            self.grads[self.n_grad - i - 1] = np.dot(delta, h.T)

            # Add regularisation
            if lambd:
                self.grads[self.n_grad - i - 1][:, :-1] += lambd * self.layers[self.n_grad - i - 1][:, :-1]

            # propagate delta to the previous layer
            temp = np.dot(delta.T, self.layers[self.n_grad - i - 1]).T[:-1, :]

            delta = np.multiply(temp, dh)

        self.cache = []

        # do computation for the input (no activation)
        try:
            input.shape[1]
        except IndexError:
            input = input[:, np.newaxis]

        self.grads[0] = np.dot(delta, input.T)
        self.grads[0][:, :-1] += lambd * self.layers[0][:, :-1]
        return

    def update(self):
        """
        Modify the weight matrix based on gradient computations
        :return:
        """
        alpha = self.alpha
        for i, grad in enumerate(self.grads):
            self.layers[i] -= alpha * grad

    def loss(self, outputs, labels):
        """
        :param output: an array outputed by softmax
        :param labels: as a float or as a batch-size list of floats
        :return: average loss
        """
        try:
            n = len(labels)
        except TypeError:
            return -np.log(outputs[labels])
        results = [-np.log(outputs[labels[i], i]) for i in range(n)]
        return sum(results) / n

    def accuracy(self, outputs, labels):
        """
        :param output: an array outputed by softmax
        :param labels: as a float
        :return: average loss
        """
        try:
            n = len(labels)
        except TypeError:
            return np.argmax == outputs[labels]
        results = np.argmax(outputs, axis=0) == labels
        return sum(results) / n

    def train(self, training_data, validation_data, epoch=10):
        """
        Train the NN
        :param training_data: data for the training
        :param validation_data: data for the validation (computing the accuracy)
        :param epoch: number of training on the training set
        :return:
        """
        batch_size = self.batch_size

        x, y = training_data
        x = np.concatenate((x, np.ones(len(x))[:, np.newaxis]), axis=1)

        # Keep a save of the best model
        best_model = copy.deepcopy(self)
        best_accuracy = self.test(validation_data)

        epoch_validation_accuracy = [best_accuracy]
        epoch_training_loss = []

        print("Initial accuracy {}".format(self.test(valid_set)))

        # now x is 785 long
        for epoch in range(epoch):  # loop over the dataset multiple times

            assert len(x) == len(y)
            n = len(x)

            running_loss = 0.0
            for i in range(0, n, batch_size):
                # get the inputs
                inputs, labels = x[i:i + batch_size].T, y[i:i + batch_size].T

                # forward + backward + optimize
                outputs = self.forward(inputs)

                loss = self.loss(outputs, labels) * batch_size
                self.backwards(inputs, outputs, labels)
                self.update()
                running_loss += loss
                if not i % 500:
                    pass
                    print('[%d, %5d] loss: %.3f    norm_W1 = %.2f' % (
                        epoch + 1, i + 1, running_loss / (i + 1), np.linalg.norm(self.layers[0])))

            epoch_training_loss.append(running_loss / n)
            current_accuracy = self.test(validation_data)
            epoch_validation_accuracy.append(current_accuracy)
            print(
                "Epoch training loss {}; accuracy {}\n".format(epoch_training_loss[-1], epoch_validation_accuracy[-1]))

            if current_accuracy > best_accuracy:
                best_model = copy.deepcopy(self)
                best_accuracy = current_accuracy

        print("Best accuracy {}".format(best_accuracy))
        return best_model, epoch_training_loss, epoch_validation_accuracy

    def test(self, test_data):
        """
        Compute the accuracy of the NN on a test set
        :param test_data: set to evaluate
        :return: accuracy
        """
        batch_size = self.batch_size

        x_test, y_test = test_data
        x_test = np.concatenate((x_test, np.ones(len(x_test))[:, np.newaxis]), axis=1)
        # now x is 785 long
        assert len(x_test) == len(y_test)
        n = len(x_test)

        test_running_loss = 0.0
        for i in range(0, n, batch_size):
            # get the inputs
            inputs, labels = x_test[i:i + batch_size].T, y_test[i:i + batch_size].T

            outputs = self.forward(inputs)
            test_loss = self.accuracy(outputs, labels) * batch_size
            test_running_loss += test_loss
        return test_running_loss / n

    def save(self, name):
        """
        Save the parameters of the NN and its layers
        :param name: name of the model, will be saved in directory : ./saved_models/
        :return:
        """
        params = [self.hidden_dims,
                  self.input_size,
                  self.output_size,
                  self.init_method,
                  self.non_linearity,
                  self.batch_size,
                  self.alpha,
                  self.lambd]
        layers = self.layers
        path = 'saved_models/' + name
        np.save(path, np.array([params, layers]))

    def load(self, name):
        """
        Load a model previously saved
        :param name: name of the model, will be loaded from directory : ./saved_models/
        :return:
        """
        path = 'saved_models/' + name
        params, layers = np.load(path)
        [self.hidden_dims,
         self.input_size,
         self.output_size,
         self.init_method,
         self.non_linearity,
         self.batch_size,
         self.alpha,
         self.lambd] = params
        self.layers = layers
        self.cache = []
        self.n_grad = len(self.layers)
        print("Parameters: {}".format(params))

    def validate_gradient(self, input, label, p=10, epsilon=0.01):
        """
        Compare the actual gradient and its finit estimation
        :param input:
        :param label:
        :param p:
        :param epsilon:
        :return:
        """
        layer_checked = 2
        output = self.forward(input)
        self.backwards(input, output, label, lambd=0)
        estimated = self.grads[layer_checked][0, :p]

        # experimental one
        experimental_grad = []
        for i in range(p):
            self.layers[layer_checked][0, i] += epsilon
            out1 = self.forward(input)
            loss1 = self.loss(out1, label)
            self.layers[layer_checked][0, i] -= 2 * epsilon
            out2 = self.forward(input)
            loss2 = self.loss(out2, label)
            self.layers[layer_checked][0, i] += epsilon

            exp_grad_i = float(loss1 - loss2) / (2 * epsilon)
            experimental_grad.append(exp_grad_i)
        self.cache = []
        return estimated, np.array(experimental_grad)


def random_search(name, duration, epoch, alpha_range=[0.001, 0.005, 0.01, 0.05, 0.1], hidden_layer_range=range(10, 784),
                  non_linearity_range=['relu', 'tanh', 'sigmoid'], batch_size_range=[4, 8, 16, 32, 64],
                  lambd_range=[0.005, 0.01, 0.05, 0.1]):
    """
    Perform a random search and save the best model
    :param name: name of the model to save
    :param duration: in seconds, duration of the search
    :param epoch: number of epochs for every single training
    :param alpha_range: range of the random parameter
    :param hidden_layer_range: range of the random parameter
    :param non_linearity_range: range of the random parameter
    :param batch_size_range: range of the random parameter
    :param lambd_range: range of the random parameter
    :return:
    """
    t0, t = time.time(), time.time()

    train_set, valid_set, test_set = np.load('data/mnist3.npy')
    best_accuracy = -1
    cmpt = 1

    while t - t0 < duration:
        print("Trial number {}".format(cmpt))

        alpha = random.choice(alpha_range)
        hidden_layer = random.sample(hidden_layer_range, 2)
        hidden_layer.sort(reverse=True)
        hidden_layer = tuple(hidden_layer)
        non_linearity = random.choice(non_linearity_range)
        batch_size = random.choice(batch_size_range)
        lambd = random.choice(lambd_range)

        random_net = NN(init_method=2, alpha=alpha, hidden_dims=hidden_layer, non_linearity=non_linearity,
                        batch_size=batch_size, lambd=lambd)

        trained_net, loss, accuracy = random_net.train(training_data=train_set, validation_data=valid_set, epoch=epoch)

        if (best_accuracy == -1) or (max(accuracy) > best_accuracy):
            trained_net.save(name)
            best_accuracy = max(accuracy)
            print("Best model: alpha {}; hidden layers {}; non linearity {}; batch size {}; lambda {}".format(
                alpha, hidden_layer, non_linearity, batch_size, lambd))

        t = time.time()
        cmpt += 1


if __name__ == '__main__':
    pass

    # Load Data
    train_set, valid_set, test_set = np.load('data/mnist3.npy')

    '''
    ### Test the neural network
    start_time = time.time()
    net = NN()
    net, _, _ = net.train(train_set, valid_set, epoch=10)
    print("Accuracy: {}".format(net.test(test_set)))
    elapsed_time = time.time() - start_time
    print('Training time = ', elapsed_time)
    '''
    '''
    ### Compare the initialization methods
    # Zero
    net = NN(init_method=0)
    zero_net, zero_training_loss, _ = net.train(train_set, valid_set)
    zero_net.save(name='zero.npy')

    # Normal
    net = NN(init_method=1)
    normal_net, normal_training_loss, _ = net.train(train_set, valid_set)
    normal_net.save(name='normal.npy')

    # Glorot
    net = NN(init_method=2)
    glorot_net, glorot_training_loss, _ = net.train(train_set, valid_set)
    glorot_net.save(name='glorot.npy')

    # Print the results
    print("Accuracy: zero {}; normal {}; glorot {}".format(zero_net.test(test_set), normal_net.test(test_set),
                                                           glorot_net.test(test_set)))
    x = range(1, 11)
    plt.plot(x, zero_training_loss, label='Zero')
    plt.plot(x, normal_training_loss, 'go', label='Normal')
    plt.plot(x, glorot_training_loss, 'r--', label='Glorot')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('Learning.pdf')
    plt.show()
    '''
    '''
    ### Random search
    # random_search(name='random_search_2', duration=4 * 3600, epoch=5)
    net = NN(save_path='random_search.npy')
    print(net.test(test_set))
    '''
    # '''
    ### Finite gradient approximation
    net = NN(save_path='glorot.npy')

    np.random.seed(8)
    input = np.randn(785, 1)
    input[-1] = 1
    label = 3

    ks = range(1, 6)
    i_range = range(6)
    eps_range = [1 / (k * 10 ** i) for k in ks for i in i_range]
    eps_range = list(reversed(sorted(eps_range)))
    log_eps = np.log(eps_range)
    res = []

    for eps in eps_range:
        estimated, experimental = net.validate_gradient(input, label, epsilon=eps, p=6)
        res.append(np.linalg.norm(experimental - estimated))
    plt.plot(log_eps, res)
    plt.xlabel('Log(epsilon)')
    plt.ylabel('Mean distance')
    plt.savefig('Finite_difference_validation.pdf')
    plt.show()
    # '''
