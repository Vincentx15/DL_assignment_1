import scipy as np
from scipy.special import softmax
import time
import matplotlib.pyplot as plt
import random
import numpy as npy


class NN:

    def __init__(self, hidden_dims=(700, 300), input_size=784, output_size=10, init_method=0,
                 non_linearity='relu', batch_size=16, lambd=0.01, save_path=None):

        if save_path is not None:
            self.load(save_path)
            self.non_linearity = 'relu'
            self.grads = list(range(self.n_grad))
            return

        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.non_linearity = non_linearity
        self.layers = self.initialise_weights()
        self.cache = []
        self.batch_size = batch_size
        self.lambd = lambd
        self.n_grad = len(self.layers)
        self.grads = list(range(self.n_grad))

    def initialise_weights(self):
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

    def activation(self, x, method):
        if method == 'relu':
            x[x < 0] = 0
            return x
        else:
            raise ValueError('Wrong method for activation')

    def forward(self, input):
        """
        :param input: should be padded
        :return:
        """

        # input is the 'h', out is the 'a' pre-activations
        for i, layer in enumerate(self.layers):
            out = np.dot(layer, input)

            if np.isnan(out).any():
                # print(out)
                print(i)
                raise ValueError('a', i)

            # The cache contains the preactivations except the input and the last one (that is fed
            # to a softmax so the computation is a bit different
            self.cache.append(out)
            input = np.concatenate((self.activation(out, self.non_linearity), np.ones((1, out.shape[1]))), axis=0)

            if np.isnan(input).any():
                # print(input)
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
            # print('final', final)
            raise ValueError('final', i)

        return final

    def backwards(self, input, output, labels, cost='cross_entropy', lambd=None):
        """
        :param input: not padded
        :param output: a list outputed by softmax
        :param labels: as a float
        :param cost: if different must change how delta_final is computed
        :return:
        """
        if lambd is None:
            lambd = self.lambd

        # vector of -out(i) except for the label where it is 1-out(i)
        if cost == 'cross_entropy':
            delta = output
            try:
                n = len(labels)
                for i in range(n):
                    delta[labels[i], i] -= 1
            except TypeError:
                delta[labels] += 1
            # print(delta, delta.shape)

        for i, a in enumerate(reversed(self.cache)):
            # compute activation again and also d(h(a))/d(a) (useful for propagation of delta)
            h = np.concatenate((self.activation(a, self.non_linearity), np.ones((1, a.shape[1]))), axis=0)
            # h = h[:, np.newaxis]

            # CAREFUL Scipy doc says sign returns -1, 1 but mine returns 0,1 ... WTF
            # dh = np.sign(a) / 2 + 0.5
            dh = np.sign(a)
            # print('a : ', np.sign(a))

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

            # To check that the gradient evolve correctly, monitor their evolution in the n_grad - l layer over passes
            # l = 0
            # if i==l:
            #     print('first grad', np.dot(delta, h.T))
            #     print('value', self.layers[self.n_grad - i - 1])
            #     print('regularised', self.grads[self.n_grad - i - 1])
            # print(self.grads[self.n_grad - i - 1].shape)
            # print('delta', delta, delta.shape)
            # print('h.T', h.T, h.T.shape)

            # propagate delta to the previous layer
            temp = np.dot(delta.T, self.layers[self.n_grad - i - 1]).T[:-1, :]

            # print('temp', temp[:3], temp.shape)
            # print('dh', dh[:3], dh.shape)

            delta = np.multiply(temp, dh)

            # print('delta', delta[:3], delta.shape)
        self.cache = []

        # do computation for the input (no activation)
        try:
            input.shape[1]
        except IndexError:
            input = input[:, np.newaxis]
        self.grads[0] = np.dot(delta, input.T)
        self.grads[0][:, :-1] += lambd * self.layers[0][:, :-1]
        return

    def update(self, alpha=0.01):
        """
        Modify the weight matrix based on gradient computations
        :param alpha: step size
        :return:
        """
        for i, grad in enumerate(self.grads):
            self.layers[i] -= alpha * grad

    def loss(self, outputs, labels):
        """
        :param output: an array outputed by softmax
        :param labels: as a float or as a batch-size list of floats
        :return: average loss
        """
        # easy to compute with this label representation, just compute the log of the proba associated with the 1
        # Just have to be careful with the size of the labels (otherwise it is not an array but
        # a single number and has no length
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
        # print(labels)
        # print(np.argmax(outputs, axis=0))
        # print(results)
        # print(sum(results) / n)
        return sum(results) / n

    def train(self, data, batch_size=None, epoch=10):
        if batch_size is None:
            batch_size = self.batch_size
        x, y = data
        x = np.concatenate((x, np.ones(len(x))[:, np.newaxis]), axis=1)

        epoch_error = []
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

                # Debugging bloc for the numerical overflows
                # try:
                #     outputs = self.forward(inputs)
                # except ValueError:
                #     self.cache = []
                #     continue
                # if np.isnan(outputs).any():
                #     print(inputs)
                #     raise ValueError('failed at {} element, epoch no {}'.format(i, epoch))

                loss = self.loss(outputs, labels) * batch_size
                self.backwards(inputs, outputs, labels)
                self.update()
                running_loss += loss
                if not i % 500:
                    pass
                    # print(self.layers[0])
                    print('[%d, %5d] loss: %.3f' % (
                        epoch + 1, i + 1, running_loss / (i + 1)))
            epoch_error.append(running_loss / n)
        return epoch_error

    def test(self, data):
        batch_size = self.batch_size
        x, y = data
        x = np.concatenate((x, np.ones(len(x))[:, np.newaxis]), axis=1)
        # now x is 785 long
        assert len(x) == len(y)
        n = len(x)

        running_loss = 0.0
        for i in range(0, n, batch_size):
            # get the inputs
            inputs, labels = x[i:i + batch_size].T, y[i:i + batch_size].T

            # forward + backward + optimize
            outputs = self.forward(inputs)
            loss = self.accuracy(outputs, labels) * batch_size
            running_loss += loss
        return running_loss / n

    def save(self, name):
        """

        :param name: name of the model, will be saved in directory : ./saved_models/
        :return:
        """
        params = [self.hidden_dims,
                  self.input_size,
                  self.output_size,
                  self.init_method,
                  self.non_linearity,
                  self.batch_size,
                  self.lambd]
        layers = self.layers
        path = 'saved_models/' + name
        np.save(path, np.array([params, layers]))

    def load(self, name):
        """

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
         self.lambd] = params
        self.layers = layers
        self.cache = []
        self.n_grad = len(self.layers)

    def validate_gradient(self, input, label, p=10, epsilon=0.01):
        layer_checked = 2
        output = self.forward(input)
        self.backwards(input, output, label, lambd=0)
        estimated = self.grads[layer_checked][0, :p]

        # experimental one
        experimental_grad = []
        # print(self.layers[0])
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
        # Check that the layers have not been modified by this procedure
        # print(self.layers[0])

        return estimated, np.array(experimental_grad)


def random_search(name, duration, epoch, hidden_layer_range=range(20, 701), non_linearity_range=['relu'],
                  batch_size_range=[4, 8, 16, 32], lambd_range=[0.005, 0.01, 0.05, 0.1]):

    t0, t = time.time(), time.time()

    train_set, valid_set, test_set = np.load('data/mnist3.npy')
    best_error = -1
    cmpt = 1

    while t-t0 < duration:
        print("Trial number {}".format(cmpt))

        hidden_layer = tuple(random.sample(hidden_layer_range, 2).sort(reverse=True))
        non_linearity = random.choice(non_linearity_range)
        batch_size = random.choice(batch_size_range)
        lambd = random.choice(lambd_range)

        random_net = NN(init_method=2, hidden_dims=hidden_layer, non_linearity=non_linearity, batch_size=batch_size,
                        lambd=lambd)

        error = random_net.train(train_set, batch_size=16, epoch=epoch)

        if (best_error == -1) or (error[-1] < best_error):
            random_net.save(name)
            best_error = error[-1]
            print("Best model: hidden layers {}; non linearity {}; batch size {}; lambda {}".format(hidden_layer, non_linearity, batch_size, lambd))

        t = time.time()
        cmpt += 1


if __name__ == '__main__':
    pass

    # Test the NN
    # net = NN(init_method=2)
    # test_in = np.randn(785, 1)
    # out = net.forward(test_in)
    # # print(out, type(out))
    # label = np.zeros((10, 1))
    # label[4] = 1
    # net.backwards(test_in, out, label)

    # Load Data
    train_set, valid_set, test_set = np.load('data/mnist3.npy')

    # Test the training procedure
    # start_time = time.time()
    # net.train(train_set, batch_size=32)
    # elapsed_time = time.time() - start_time
    # print('CPU time = ', elapsed_time)

    # Test the training procedure
    # acc = net.test(valid_set)
    # print(acc)
    # 40s for 10 000 pass without vectorisation
    # 4s with batch size = 16 for the same number also better results


    # Initialization
    '''
    # net = NN(init_method=0)
    # zero = net.train(train_set, batch_size=32)
    x = range(1, 11)
    zero = [2.303 for i in x]

    net = NN(init_method=1, lambd=0.1)
    normal = net.train(train_set, batch_size=16)
    net.save(name='normal.npy')
    # Final loss : 0.153


    net = NN(init_method=2, lambd=0.1)
    glorot = net.train(train_set, batch_size=16)
    net.save(name='glorot.npy')
    # Final loss : 0.164

    print(glorot)
    print(normal)
    print(zero)
    # [0.3313110359219607, 0.278203561038047, 0.27245356025651, 0.270768605866088, 0.27021384828256084, 0.2698946275128716, 0.2699768048668205, 0.27008109747934655, 0.2701660349959337, 0.27084915005548876]
    # [0.33634069937246813, 0.278272198137196, 0.27254340211815475, 0.27030638072299945, 0.269998828237002, 0.2704305975279742, 0.2710364760484686, 0.2709348648177925, 0.27110026319298175, 0.2716033660764445]
    # [2.303, 2.303, 2.303, 2.303, 2.303, 2.303, 2.303, 2.303, 2.303, 2.303]

    plt.plot(x, zero, label='Zero')
    plt.plot(x, normal, label='Normal')
    plt.plot(x, glorot, label='Glorot')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.savefig('Learning.pdf')
    plt.show()
    '''


    '''
    # test saving module
    net = NN(init_method=2, lambd=0.3)
    net.train(train_set, batch_size=32)
    net.save(save_path='test.npy')
    net = NN(save_path='test.npy')
    print(net.layers)
    net.train(train_set)
    '''


    '''
    # Gradient Validation
    # x, y = train_set
    # x = np.concatenate((x, np.ones(len(x))[:, np.newaxis]), axis=1)
    # k = random.randint(0, 150)
    # input, label = x[k][:, np.newaxis], y[k]
    net = NN(save_path='glorot.npy')

    np.random.seed(8)
    input = np.randn(785, 1)
    input[-1] = 1
    label = 3
    # estimated, experimental = net.validate_gradient(input, label, epsilon=0.1, p=6)
    # print(estimated, experimental)

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
    plt.legend()
    plt.savefig('Finite_difference_validation.pdf')
    plt.show()
    '''