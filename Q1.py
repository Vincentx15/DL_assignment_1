import scipy as np
from scipy.special import logsumexp
import time


def softmax(x, axis):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


class NN():

    def __init__(self, hidden_dims=(700, 300), input_size=784, output_size=10, init_method=0,
                 non_linearity='relu', batch_size=16):
        self.hidden_dims = hidden_dims
        self.input_size = input_size
        self.output_size = output_size
        self.init_method = init_method
        self.non_linearity = non_linearity
        self.layers, self.grads = self.initialise_weights(self.hidden_dims)
        self.n_grad = len(self.layers)
        self.cache = []
        self.batch_size = batch_size

    def initialise_weights(self, hidden_dims):
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
                return np.randn(shape)
            else:
                d = np.sqrt(6.0 / np.sum(shape))
                return np.random.uniform(low=-d, high=d, size=shape)

        dims = (self.input_size, *hidden_dims, self.output_size)
        layers = []
        grads = []
        for i in range(len(dims) - 1):
            shape = (dims[i + 1], dims[i] + 1)
            layer = create_shape(shape)
            grad = create_shape(shape, method=0)
            layers.append(layer)
            grads.append(grad)
        return layers, grads

    def activation(self, input, method):
        if method == 'relu':
            input[input < 0] = 0
            return input
        else:
            raise ValueError('Wrong method for activation')

    def forward(self, input):
        """
        :param input: should be padded
        :return:
        """

        # input is the 'h', out is the 'a' preactivations
        for i , layer in enumerate(self.layers):
            out = np.dot(layer, input)

            if np.isnan(out).any():
                print(out)
                raise ValueError('a', i)

            # The cache contains the preactivations except the input and the last one (that is fed
            # to a softmax so the computation is a bit different
            self.cache.append(out)
            input = np.concatenate((self.activation(out, self.non_linearity), np.ones((1, out.shape[1]))), axis=0)
            if np.isnan(input).any():
                print(input)
                raise ValueError('input', i)
            # Check if shape is not such as (25,), should not happen when using batch, but could happen with single pass
            try:
                input.shape[1]
            except IndexError:
                input = input[:, np.newaxis]

        # remove last activation from cache
        self.cache.pop()
        final = softmax(out, axis=0)
        if np.isnan(final).any():
            print('out', out)
            print('final', final)
            raise ValueError('final', i)
        return final

    def backwards(self, input, output, labels, cost='cross_entropy'):
        """
        :param input: not padded
        :param output: a list outputed by softmax
        :param labels: as a float
        :param cost: if different must change how delta_final is computed
        :return:
        """
        # vector of -out(i) except for the label where it is 1-out(i)
        if cost == 'cross_entropy':
            delta = - output
            try:
                n = len(labels)
                for i in range(n):
                    delta[labels[i], i] += 1
            except TypeError:
                delta[labels] += 1
            # print(delta, delta.shape)

        for i, a in enumerate(reversed(self.cache)):
            # compute activation again and also d(h(a))/d(a) (useful for propagation of delta
            h = np.concatenate((self.activation(a, self.non_linearity), np.ones((1, a.shape[1]))), axis=0)
            # h = np.append(self.activation(a, self.non_linearity), 1)
            # h = h[:, np.newaxis]
            dh = np.sign(a) / 2 + 0.5

            # sign sometimes don't add dimension sometimes yes
            try:
                dh.shape[1]
            except IndexError:
                dh = dh[:, np.newaxis]

            # compute grads
            self.grads[self.n_grad - i - 1] = np.dot(delta, h.T)
            # print(self.grads[self.n_grad - i - 1].shape)
            # print('delta', delta, delta.shape)
            # print('h.T', h.T, h.T.shape)

            # propagate delta to the previous layer
            temp = np.dot(delta.T, self.layers[self.n_grad - i - 1]).T[:-1, :]
            delta = np.multiply(temp, dh)

            # print('temp',temp, temp.shape)
            # print('dh',dh, dh.shape)
            # print('delta',delta, delta.shape)
        self.cache = []

        # do computation for the input (no activation)
        try:
            input.shape[1]
        except IndexError:
            input = input[:, np.newaxis]
        self.grads[0] = np.dot(delta, input.T)
        return

    def update(self, alpha=0.01):
        """
        Modify the wieght matrix based on gradient computations
        :param alpha: step size
        :return:
        """
        for i, grad in enumerate(self.grads):
            self.layers[i] += alpha * grad

    def loss(self, outputs, labels):
        """
        :param output: an array outputed by softmax
        :param labels: as a float
        :return: average loss
        """
        # easy to compute with this label representation, just compute the log of the proba associated with the 1
        # Just have to be careful with the size of the labels (otherwise it is not an array but
        # a single number and has no length
        try:
            n = len(labels)
        except TypeError:
            return -np.log(outputs[labels])
        results = [-np.log(0.00000000000001 + outputs[labels[i], i]) for i in range(n)]
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
        # now x is 785 long
        for epoch in range(epoch):  # loop over the dataset multiple times
            assert len(x) == len(y)
            n = len(x)

            running_loss = 0.0
            for i in range(0, n, batch_size):
                # get the inputs
                inputs, labels = x[i:i + batch_size].T, y[i:i + batch_size].T

                # forward + backward + optimize
                try :
                    outputs = self.forward(inputs)
                except ValueError:
                    self.cache = []
                    continue
                # if np.isnan(outputs).any():
                #     print(inputs)
                #     raise ValueError('failed at {} element, epoch no {}'.format(i, epoch))
                loss = self.loss(outputs, labels) * batch_size
                self.backwards(inputs, outputs, labels)
                self.update()
                running_loss += loss
                if not i % 500:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))

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


if __name__ == '__main__':
    pass
    net = NN(init_method=2)

    # test_in = np.randn(785, 1)
    # out = net.forward(test_in)
    # # print(out, type(out))
    # pred = out
    # label = np.zeros((10, 1))
    # label[4] = 1
    # net.backwards(test_in, pred, label)

    # np.save(open('data/mnist3' + '.npy', 'wb'), (train_set, valid_set, test_set))
    train_set, valid_set, test_set = np.load('data/mnist3.npy')

    start_time = time.time()
    net.train(train_set, batch_size=32)

    elapsed_time = time.time() - start_time
    print('CPU time = ', elapsed_time)

    acc = net.test(valid_set)
    print(acc)
    # 40s for 10 000 pass without vectorisation
    # 4s with batch size = 16 for the same number also better results
