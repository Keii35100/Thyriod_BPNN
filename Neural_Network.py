# Imports
import numpy as np
import pandas as pd

# Read dataset
data = pd.read_csv('cl_thyroid_conditions.csv')

# Converting class label to binary
data["target_sick"] = data["target_sick"].astype('category')
data["target_sick_bin"] = data["target_sick"].cat.codes
data["sex=M"] = data["sex=M"].astype(float)

# Assign input and output
X = np.array(data.drop(['target_sick', 'target_sick_bin'], axis=1), dtype=float).T
y = np.array(np.split(data['target_sick_bin'], len(data['target_sick_bin'])), dtype=float).T

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.weights1= np.random.rand(516,7) # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(7,516)
        self.y = y
        self.output = np.zeros(y.shape)
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2
        
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (self.y - self.output)*sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot((self.y -self.output)*sigmoid_derivative(self.output), self.weights2.T)*sigmoid_derivative(self.layer1))
    
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        

NN = NeuralNetwork(X,y)
for i in range(100): # trains the NN 1,000 times
        print ("for iteration # " + str(i) + "\n")
        #print ("Input : \n" + str(X))
        #print ("Actual Output: \n" + str(y))
        #print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("Percentage: \n" + str(100 - (np.mean(np.square(y - NN.feedforward()))).astype(float))) # mean sum squared loss
        print ("\n")
  
NN.train(X, y)
