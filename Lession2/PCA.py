from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

class Kernel(object):
    def __init__(self):
        pass

    def algorithms(self):
        digits = load_digits()
        data = digits.data
        print(data.shape)
        # mnist = fetch_mldata('MNIST original', data_home='/home/tuanlv/PycharmProjects/TrainingML/Day2')
        # print(mnist.data.shape)
        pca = PCA(n_components=15, whiten=False)
        data = pca.fit_transform(digits.data)

        # use grid search cross-validation to optimize the bandwidth
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params)
        grid.fit(data)

        print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

        # use the best estimator to compute the kernel density estimate
        kde = grid.best_estimator_

        # sample 44 new points from the data
        new_data = kde.sample(44, random_state=0)
        new_data = pca.inverse_transform(new_data)

        # turn data into a 4x11 grid
        new_data = new_data.reshape((4, 11, -1))
        real_data = digits.data[:44].reshape((4, 11, -1))

        # plot real digits and resampled digits
        fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
        for j in range(11):
            ax[4, j].set_visible(False)
            for i in range(4):
                im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)),
                                     cmap=plt.cm.binary, interpolation='nearest')
                im.set_clim(0, 16)
                im = ax[i + 5, j].imshow(new_data[i, j].reshape((8, 8)),
                                         cmap=plt.cm.binary, interpolation='nearest')
                im.set_clim(0, 16)

        ax[0, 5].set_title('Selection from the input data')
        ax[5, 5].set_title('"New" digits drawn from the kernel density model')

        plt.show()

class TestPCA(object):
    def __init__(self, mnist):
        self.mnist = mnist

    def algorithms(self):
        pca = PCA(n_components=324)
        print(pca)
        pca.fit(self.mnist.data)
        U = pca.components_.T

        real_data = self.mnist.data
        fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
        for j in range(11):
            x = real_data[j].reshape(784, 1) - pca.mean_.reshape(784, 1)
            z = U.T.dot(x)
            new_data = U.dot(z) + pca.mean_.reshape(784, 1)
            ax[1, j].set_visible(False)
            for i in range(1):
                im = ax[i, j].imshow(real_data[j].reshape((28, 28)),
                                     cmap=plt.cm.binary, interpolation='nearest')
                im.set_clim(0, 16)
                im = ax[i + 2, j].imshow(new_data.reshape((28, 28)),
                                         cmap=plt.cm.binary, interpolation='nearest')
                im.set_clim(0, 16)

        ax[0, 2].set_title('Origional Data')
        ax[2, 2].set_title('Reconstruction Data')
        plt.show()

class DemoPCA(object):
    def __init__(self, data):
        self.data = data

    def algorithms(self):
        pca = PCA(n_components=0.99, svd_solver='full')
        data = np.array(self.data)
        print(pca)
        pca.fit(data)
        U = pca.components_.T

        fig = plt.figure(figsize=(8,8))

        for i in range(1, 11):
            pixels = data[i]
            pixels = np.array(pixels, dtype='uint8')
            pixels = pixels.reshape((28, 28))
            fig.add_subplot(4, 10, i)
            f0 = plt.imshow(pixels, cmap='gray')
            f0.axes.get_xaxis().set_visible(False)
            f0.axes.get_yaxis().set_visible(False)

            x = data[i].reshape(784, 1) - pca.mean_.reshape(784, 1)
            z = U.T.dot(x)
            x_tilde = U.dot(z) + pca.mean_.reshape(784, 1)
            fig.add_subplot(4, 10, i + 10)
            f1 = plt.imshow(x_tilde.reshape(28, 28), cmap='gray')
            f1.axes.get_xaxis().set_visible(False)
            f1.axes.get_yaxis().set_visible(False)

            fig.add_subplot(4, 10, i + 20)
            f2 = plt.imshow(U[:, i].reshape(28, 28), cmap='gray')
            f2.axes.get_xaxis().set_visible(False)
            f2.axes.get_yaxis().set_visible(False)

            mean = np.mean(data)
            x_n = data[i] - mean
            z_n = U.T.dot(x_n)
            y_n = U.dot(z_n) + mean
            fig.add_subplot(4, 10, i + 30)
            f3 = plt.imshow(y_n.reshape(28, 28), cmap='gray')
            f3.axes.get_xaxis().set_visible(False)
            f3.axes.get_yaxis().set_visible(False)
        plt.gray()
        plt.axis('off')
        plt.show()

class BuildPCA(object):
    def __init__(self, data, n_components):
        self.data = data
        self.n_components = n_components

    def __algorithms_small_sample(self):
        data = np.array(self.data)
        data = data - np.mean(data)
        #S = data.dot(data.T)
        #lamda, vecto
        #eigval, eigvec = np.linalg.eig(S)
        #PCA
        #order = np.argsort(eigval)[::-1]
        #components = eigvec[:, order[:self.n_components]]
        U, S, V = np.linalg.svd(data, full_matrices=False)
        components = V[:self.n_components]
        #projection
        z = data.dot(components.T)
        return z, components

    def __algorithms_large_sample(self):
        data = np.array(self.data)
        data = data - np.mean(data)
        L = data.dot(data.T)
        eigval, eigvec = np.linalg.eig(L)
        #eigvec = data.dot(eigvec)
        order = np.argsort(eigval)[::-1]
        components = eigvec[:,order[:self.n_components]]
        z = data.dot(components)
        return z, components

    def get_n_component(self, ratio, eigval):
        '''
        accuracy of data after PCA
        '''
        val = eigval[np.argsort(eigval)]
        sum = 0
        for i in range(len(val)):
            sum += val[i]
            if sum > ratio:
                return i
        return 0

    def algorithms(self, type='1'):
        if type == '1':
            z, components = self.__algorithms_small_sample()
        else:
            z, components = self.__algorithms_large_sample()

        data = np.array(self.data)
        mean = np.mean(data)
        fig = plt.figure(figsize=(8, 8))
        for i in range(1,11):
            pixels = data[i]
            pixels = np.array(pixels, dtype='uint8')
            pixels = pixels.reshape((28, 28))
            fig.add_subplot(2, 10, i)
            f0 = plt.imshow(pixels, cmap='gray')
            f0.axes.get_xaxis().set_visible(False)
            f0.axes.get_yaxis().set_visible(False)

            x_n = data[i] - mean
            z_n = components.dot(x_n)
            y_n = components.T.dot(z_n) + mean
            fig.add_subplot(2, 10, i + 10)
            f1 = plt.imshow(y_n.reshape(28,28), cmap='gray')
            f1.axes.get_xaxis().set_visible(False)
            f1.axes.get_yaxis().set_visible(False)
        plt.show()

if __name__ == '__main__':
    mnist = fetch_mldata('MNIST original', data_home='/home/tuanlv/PycharmProjects/TrainingML/Day2')
    print(mnist.data.shape)

    DemoPCA(mnist.data).algorithms()
    #Kernel().algorithms()
    #TestPCA(mnist).algorithms()
    #BuildPCA(mnist.data[:1000,:], 59).algorithms(type='1')






