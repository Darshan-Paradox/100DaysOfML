## Importing necessary libraries
import numpy as np
from matplotlib import pyplot as plt

## Arguments for Linear regression model

## Batch_size if > length of output matrix or smaller than 1
## will perform batch gradient otherwise for size > 1 but less than length
## it will perform mini-batch gradient, and for size == 1 it will perform
## stochastic gradient

## Regularisation takes on 2 inputs L1 or L2
## It implements respective regularisation in model to prevent overfitting
## Lambda is the parameter for the amount of regularisation needed.

## Plot takes boolean value as input, if True it will plot cost vs iterations

ARGS = {"batch_size": 0,
        "regularisation": None,
        "lambda": 0,
        "plots": True}

## Declaring and defining Model class for Linear Regression

class Model:
    def __init__(self, lr, epochs, args = ARGS):
        self.lr = lr # learning rate
        self.epochs = epochs # number of iterations
        self.lamda = args["lambda"] # regularisation constant

        self.batch_size = args["batch_size"]
        self.reg = args["regularisation"] # regularisation method
        self.plots = args["plots"] # plots

    ## Training function takes training input and output as parameters and
    ## trains the model using respective gradient descent methods and regularisations
    
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
            
            ## L is regularisation term in cost function
            ## _L is regularisation term in gradient descent term

            L = 0
            _L = 0
            
            ## _X is mini-batch of X, it could either be equal to X or a subset of it
            ## _E is mini-batch of E, it could either be equal to E or a subset of it
            
            _X = X
            _E = E
            
            ## Making mini-batch out of X and E
            
            if self.batch_size > 0 and self.batch_size < len(E):
                index = np.random.randint(0, X.shape[0], self.batch_size)
                
                _X = X[[index]]
                _E = E[[index]]

            ## Defining regularisation terms, if regularisation argument is passed in model
            
            if self.reg == "L1":
                L = self.lamda * np.sum(np.abs(W))
                _L = self.lamda * np.sign(W)
            elif self.reg == "L2":
                L = self.lamda * np.sum(np.square(W))
                _L = self.lamda * W

            ## Cost function
            
            cost.append((1/(2*len(_E)))*(_E.T.dot(_E) + L).item())

            ## Gradient descent
            
            self.W = self.W + self.lr*(1/len(_E))*(_X.T.dot(_E)) - self.lr*_L
            self.B = self.B + self.lr*(1/len(_E))*np.sum(_E, axis=0)

        return cost

    ## Test function takes test dataset as input and outputs summary of error rate and
    ## predicted outputs
    
    def test(self, X, Y):
        H = X.dot(self.W) + self.B
        summary = self.__test_summary(H, Y)

        return H, summary
    
    ## Different error parameters are defined in __test_summary and is internally used
    ## in test function
    
    def __test_summary(self, H, Y):
        MAE = np.abs(Y - H).mean()
        MSE = np.sqrt(np.square(Y - H).mean())
        R_2 = 1 - ((Y - H).var()/ Y.var())
        
        summary = {"MAE": MAE,
                  "MSE": MSE,
                  "R_2": R_2}
        
        print(f"Mean Absolute Error is {MAE * 100}%\nMean Square Error is {MSE * 100}%\nR Square Coefficient is {R_2 * 100}%\n")
        
        return summary