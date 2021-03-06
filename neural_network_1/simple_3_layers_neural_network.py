# Description : Consider trying to predict the output column given the three input columns.
# We could solve this problem by simply measuring statistics between the input values and the output values.
# If we did so, we would see that the leftmost input column is perfectly correlated with the output.
# Backpropagation, in its simplest form, measures statistics like this to make a model.
# Let's jump right in and use it to do this.

# Source : http://iamtrask.github.io/2015/07/12/basic-python-network/
from time import sleep

import numpy as np
from pip._vendor.msgpack.fallback import xrange

# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# output dataset
y = np.array([[1],
			  [0],
			  [1],
			  [1]])

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1



for iter in xrange(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # как сильно мы ошиблись относительно нужной величины?
    l2_error = y - l2


    # в какую сторону нужно двигаться?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # как сильно значения l1 влияют на ошибки в l2?
    l1_error = l2_delta.dot(syn1.T)

    # в каком направлении нужно двигаться, чтобы прийти к l1?
    # если мы были уверены в предсказании, то сильно менять его не надо
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

    sleep(0.00001)
    print('\n' * 10)
    print(l1)


print("Input dataset:")
print(l0)

print("Output dataset:")
print(y)

print("Output layer 1 Training:")
print(l1)

print("Output After Training2:")
print(l2)


