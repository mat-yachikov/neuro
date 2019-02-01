import numpy as np

x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0, 1, 1, 0]]).T


np.random.seed(1)
weights1 = np.random.random((3, 2)) - 1
weights2 = np.random.random((2, 1)) - 1


def nonlin(x, deriv=False):
    if(deriv is True):
        return x*(1-x)

    return 1/(1+np.exp(-x))


for iter in range(900000):
    z2 = np.dot(x, weights1)

    a2 = nonlin(z2)

    z3 = np.dot(a2, weights2)

    a3 = nonlin(z3)

    error = y - a3

    delta3 = error * nonlin(a3, deriv=True)
    l1error = delta3.dot(weights2.T)
    delta2 = l1error * nonlin(a2, deriv=True)

    weights2 += np.dot(a2.T, delta3)
    weights1 += np.dot(x.T, delta2)

print(a3)
