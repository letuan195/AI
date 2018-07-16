import numpy as np
import matplotlib.pyplot as plt

class GradientDescent(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = self.X.shape[0]

    def func(self, a):
        '''
        cost function
        :param a: coef
        :return: value cost
        '''
        return 1/N * (self.y - self.X.dot(a)).T.dot((self.y - self.X.dot(a)))

    def gradient(self, a):
        return 2/N * self.X.T.dot(self.X.dot(a) - self.y)

    def gradient_descent(self, a_0, eta):
        a = [a_0]
        t = 1
        while np.linalg.norm(self.gradient(a[-1])) / len(a[-1]) > 1e-3:
            a_new = a[-1] - eta*self.gradient(a[-1])
            a.append(a_new)
            t = t + 1

        # for i in range(100):
        #     h = self.gradient(a[-1])
        #     print('a[-1]: {0}, h: {1}'.format(a[-1], h))
        #     a_new = a[-1] - eta * h
        #     a.append(a_new)
        #     t = t + 1

        cost = self.func(a[-1])
        print('coef: ', a[-1])
        print('number of loop: ', t)
        print('optimal cost:', cost)
        return (a, t)

    def find_learning_rate(self, a_0, eta):
        alpha = 1 / 3
        beta = 1 / 2
        a = [a_0]
        cost1 = 1
        cost2 = 0
        t = 0
        while cost1 > cost2 and t < 1000000:
            a_new = a[-1] - eta * self.gradient(a[-1])
            cost1 = self.func(a_new)
            grad = self.gradient(a[-1])
            e_grad = np.dot(grad.T, grad)
            cost2 = self.func(a[-1]) - alpha * eta * e_grad
            a.append(a_new)
            eta = eta * beta
            t = t + 1
        return eta

class StochasticGD(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.N = self.X.shape[0]
    def func(self, a):
        return 1/N * (self.y - self.X.dot(a)).T.dot((self.y - self.X.dot(a)))

    def gradient(self, a, i, shuffle_id):
        id = shuffle_id[i]
        xi = self.X[id, :].reshape((1,2))
        yi = self.y[id].reshape((1,1))
        return 2/N * xi.T.dot(xi.dot(a) - yi)

    def algorithms(self, a_0, eta):
        a = [a_0]
        shuffle_id = np.random.permutation(N)
        for i in range(N):
            a_new = a[-1] - eta * self.gradient(a[-1], i, shuffle_id)
            a.append(a_new)

        cost = self.func(a[-1])
        print('coef: ', a[-1])
        print('optimal cost:', cost)
        return a

class MinibatchGD(object):
    def __init__(self, X, y, S):
        self.X = X
        self.y = y
        self.N = self.X.shape[0]
        self.S = S

    def func(self, a):
        return 1/N * (self.y - self.X.dot(a)).T.dot((self.y - self.X.dot(a)))

    def gradient(self, a, i):
        sum = 0
        i0 = i
        while i < self.X.shape[0] and i < (i0 + self.S):
            xi = self.X[i, :].reshape((1, 2))
            yi = self.y[i].reshape((1, 1))
            sum = sum + 2 / N * xi.T.dot(xi.dot(a) - yi)
            i = i + 1
        return sum/(i - i0)

    def algorithms(self, a_0, eta):
        a = [a_0]
        for i in range(0, N, self.S):
            a_new = a[-1] - eta * self.gradient(a[-1], i)
            a.append(a_new)

        cost = self.func(a[-1])
        print('coef: ', a[-1])
        print('optimal cost:', cost)

if __name__ == '__main__':

    N = 100
    X = np.asarray([i for i in range(1, N+1)])
    X = X.reshape((N, 1))
    y = 3 + 2 * X  + np.random.normal(0, 1, (N,1)) #random noise with mean = 0, standard = 1

    one = np.ones((N, 1))
    X_0 = np.concatenate((one, X), axis = 1) # add bias for X

    #plt.plot(X.T, y.T, 'b.')
    #plt.axis([0, 100, 0, 500])
    #plt.show()

    a_0 = np.asarray([[2], [1]]) # coef initial

    learning_rate = 0.01 # 0.0001
    gd = GradientDescent(X_0, y)
    rate = gd.find_learning_rate(a_0, learning_rate)
    print(rate)
    #a = gd.gradient_descent(a_0, rate)

    #stochastic gradient descent
    #learning_rate = 0.001
    #s_gd = StochasticGD(X_0, y)
    #a = s_gd.algorithms(a_0, learning_rate)

    #mini-batch gradient descent
    #S = 5
    #learning_rate = 0.003
    #mini_gd = MinibatchGD(X_0, y, S)
    #a = mini_gd.algorithms(a_0, learning_rate)









