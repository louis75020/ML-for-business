import scipy.stats
import numpy as np

# X : data matrix (pandas.DataFrame, np.array 2d...), dim n*p
# n_randomFeatures : 1 <= int <= p, number of random features to select
# gamma : float (parameter for the radial kernel used in SVM)


class RandomKernelFeatures():

    def __init__(self, n_randomFeatures, gamma):
        self.gamma = gamma
        self.c = n_randomFeatures

    def fit(self, X_train):
        n, p = X_train.shape
        W = scipy.stats.norm.rvs(loc = 0, scale = 2 * self.gamma, size = p * self.c)
        W = np.reshape(W, (p, self.c))
        b = scipy.stats.uniform.rvs(loc = 0, scale = 2 * np.pi, size = self.c)
        self.W = W
        self.b = b

    def transform(self, X):
        return np.sqrt(2/self.c) * np.cos(np.dot(X, self.W) + self.b)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def __main__():
    test = np.random.rand(4,4)
    print(RandomKernelFeatures(3, 1).fit_transform(test))

if __name__ == "__main__":__main__()