import numpy as np
from matplotlib import pyplot as plt

ARGS = {"gradient_descent": "BGD",
        "regularisation": None,
        "lambda": 0,
        "plots": True}

class Model:
    def __init__(self, lr, epochs, args = ARGS):
        self.lr = lr # learning rate
        self.epochs = epochs # number of iterations
        self.lamda = args["lambda"] # regularisation constant

        self.gd = args["gradient_descent"] # gradient descent method
        self.reg = args["regularisation"] # regularisation method
        self.plots = args["plots"] # plots

    def train(self, X, Y):
        self.W = np.random.rand(X.shape[-1], 1)
        self.B = np.random.rand(1, 1)

        cost = self.__grad_descent(X, Y)

        if self.plots:
            plt.plot(cost)

        return X.dot(self.W) + self.B

    def __grad_descent(self, X, Y):
        cost = []

        for i in range(self.epochs):
            H = X.dot(self.W) + self.B
            E = Y - H

            L = 0
            _L = 0
            
            if self.gd == "mgd":
                indices = self.__mini_batch_indices(0.7)
                
                X = self.__mini_batch(indices, X)
                E = self.__mini_batch(indices, E)
                
            elif self.gd == "sgd":
                index = np.random.randint(len(E));
                
                E = np.array([E[index]])
                X = np.array([X[index]])

            if self.reg == "L1":
                L = self.lamda * np.sum(np.abs(W))
                _L = self.lamda * np.sign(W)
            elif self.reg == "L2":
                L = self.lamda * np.sum(np.square(W))
                _L = self.lamda * W

            cost.append((1/(2*len(Y)))*(E.T.dot(E) + L).item())

            self.W = self.W + self.lr*(1/len(Y))*(X.T.dot(E)) - self.lr*_L
            self.B = self.B + self.lr*(1/len(Y))*np.sum(E)

        return cost
    
    def __mini_batch_indices(self, frac):
        indices = np.random.permutation(A.shape[0])
        indices = indices[:int(frac*len(indices))]
        
        return indices
    
    def __mini_batch(self, indices, A):
        dummy = []
        for i in indices:
            dummy.append(A[i])
        
        return np.array(dummy)

    def test(self, X, Y):
        return X.dot(self.W) + self.B
