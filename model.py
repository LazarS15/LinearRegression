import numpy as np


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
            Simple implementation of Linear Regression using gradient descent.

            Attributes:
                lr (float): Learning rate for gradient descent.
                n_iter (int): Number of iterations for training.
                w (numpy.ndarray): Weights for the model.
                b (float): Bias term for the model.
        """
        self.lr = learning_rate
        self.n_iter = n_iter
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        """
            Fit the model to the data using gradient descent.

            Args:
                X (numpy.ndarray): Feature matrix (n_samples, n_features).
                y (numpy.ndarray): Target values (n_samples).
        """

        n_rows, n_columns = X.shape
        self.w = np.zeros(n_columns)

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.w) + self.b

            dw = (2 / n_rows) * np.dot(X.T, (y_pred - y))
            db = (2 / n_rows) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b