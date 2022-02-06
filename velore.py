import numpy as np

activation_functions = {
    'sigmoid': lambda z : 1/(1 + np.exp(-z)),
    'ReLU': lambda z : np.maximum(.0, z)        #lr=.005 iterations=5000
}

class Velore:
    def __init__(self, learning_rate=0.01, iterations=2000, activation='sigmoid'):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation = activation_functions[activation]

    def __loss(self, y, a):
        return y*np.log(a) + (1-y)*np.log(1-a)

    def __forward_prop(self, X, Y):
        m = X.shape[1]

        A = self.activation(np.dot(self.weights.T,X) + self.bias)
        cost = (-1/m)*np.sum(self.__loss(Y, A))

        dweight = (1/m)*np.dot(X,(A - Y).T)
        dbias = (1/m)*np.sum(A - Y)
        cost = np.squeeze(np.array(cost))
        
        return dweight, dbias, cost

    def __backward_prop(self, dweight, dbias):
        self.weights = self.weights - self.learning_rate*dweight
        self.bias = self.bias - self.learning_rate*dbias

    def train(self, init_weigths, init_bias, X, Y):
        self.weights = init_weigths
        self.bias = init_bias
        costs = []
        
        #gradient descent
        for i in range(self.iterations):
            dweight, dbias, cost = self.__forward_prop(X, Y)
            self.__backward_prop(dweight, dbias)
            
            if i % 100 == 0:
                costs.append(cost)

        return costs
        

    def predict(self, X):    
        m = X.shape[1]
        weights = self.weights.reshape(X.shape[0], 1)

        A = self.activation(np.dot(weights.T, X)+self.bias)
    
        return np.where(A > .5, 1, 0)