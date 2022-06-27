import numpy as np
from matplotlib import pyplot as plt

ARGS = {"batch_size": None,
        "regularisation": None,
        "lambda": 0,
        "plots": True}

class Model:
    def __init__(self, lr, epochs, args = ARGS):
        self.lr = lr # learning rate
        self.epochs = epochs # number of iterations
        self.lamda = args["lambda"] # regularisation constant

        self.batch_size = args["batch"]
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
            
            _X = X
            _E = E
            
            if self.batch_size > 0 and self.batch_size < len(E):
                index = np.random.randint(0, X.shape[0], self.batch_size)
                
                _X = X[[index]]
                _E = E[[index]]

            if self.reg == "L1":
                L = self.lamda * np.sum(np.abs(W))
                _L = self.lamda * np.sign(W)
            elif self.reg == "L2":
                L = self.lamda * np.sum(np.square(W))
                _L = self.lamda * W

            cost.append((1/(2*len(_E)))*(_E.T.dot(_E) + L).item())

            self.W = self.W + self.lr*(1/len(_E))*(_X.T.dot(_E)) - self.lr*_L
            self.B = self.B + self.lr*(1/len(_E))*np.sum(_E, axis=0)

        return cost

    def test(self, X, Y):
        H = X.dot(self.W) + self.B
        summary = self.test_summary(X, Y)
        return H, summary
    
    def test_summary(X, Y):
        MAE = np.abs(Y - (X.dot(self.W) + self.B)).mean()
        MSE = np.square(Y - (X.dot(self.W) + self.B)).mean()
        R_2 = 0
        
        summary = {"MAE": MAE,
                  "MSE": MSE,
                  "R_2": R_2}
        
        return summary