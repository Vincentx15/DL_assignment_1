import gzip
import numpy as np
import six
from six.moves import cPickle as cpickle

'''
Pickle has changed from python 2 to python 3
'''

# Load the dataset
# with gzip.open('data/mnist.pkl.gz', 'rb') as f:
#     train_set, valid_set, test_set = cpickle.load(f, encoding='bytes')
# print(train_set)

# # np.save(open('data/mnist3' + '.npy', 'wb'), (train_set, valid_set, test_set))
# train_set, valid_set, test_set = np.load('data/mnist3.npy')
# print(train_set)

x = np.array([[0,0,1],[0,1,0],[0,0,1]])
a = np.nonzero(x)
print(a)