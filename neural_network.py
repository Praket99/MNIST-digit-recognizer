import numpy as np

class NeuralNetwork:

    def __init__(self):
        #layer 1
        #784 to 128
        self.W1 = np.random.randn(784, 128) * np.sqrt(2 / 784)
        self.b1 = np.zeros(128)

        #layer 2
        #128 to 128
        self.W2 = np.random.randn(128, 128) * np.sqrt(2 / 128)
        self.b2 = np.zeros(128)

        #128 to 10 (output layer)
        self.W3 = np.random.randn(128, 10) * np.sqrt(2 / 128)
        self.b3 = np.zeros(10)

    #ReLU activation function

    def relu(self, x):

        return np.maximum(0, x)

    #ReLU derivative function

    def relu_derivative(self, x):

        return x > 0

    #softmax 

    def softmax(self, x):

        x = x - np.max(x)

        exp_x = np.exp(x)

        return exp_x / np.sum(exp_x)

    #forward propogation

    def forward(self, x):

        #layer 1 
        self.z1 = x @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)

        #layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)

        #layer 3 output layer
        self.z3 = self.a2 @ self.W3 + self.b3
        self.a3 = self.softmax(self.z3)

        return self.a3

    #loss function

    def loss(self, prediction, label):
        
        return -np.log(prediction[label] + 1e-8)

    #backpropogation

    def backward(self, x, label):

        #output layer
        dz3 = self.a3.copy()
        dz3[label] -= 1

        #gradient for W3 and b3

        dW3 = np.outer(self.a2, dz3)

        db3 = dz3

        #layer 2

        da2 = self.W3 @ dz3

        dz2 = da2 * self.relu_derivative(self.z2)

        dW2 = np.outer(self.a1, dz2)

        db2 = dz2

        #layer 1

        da1 = self.W2 @ dz2

        dz1 = da1 * self.relu_derivative (self.z1)

        dW1 = np.outer(x, dz1)

        db1 = dz1

        return (dW1, db1, dW2, db2, dW3, db3)

    #stochastic gradient descent

    def update(self, dW1, db1, dW2, db2, dW3, db3, learning_rate):

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3