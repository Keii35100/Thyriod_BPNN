# imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# read dataset
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/cl_thyroid_conditions_2_normalized.csv')

# change class label to binary
data["target_sick"] = data["target_sick"].astype('category')
data["target_sick_bin"] = data["target_sick"].cat.codes
data["sex=M"] = data["sex=M"].astype(float)

# suffle data
cl_data = np.array(data.drop(['target_sick'], axis=1), dtype = float)
m, n = cl_data.shape
np.random.shuffle(cl_data)                          # shuffle data

persentage_split = 80/100                           # percentage split

# assign input and output
train_data = cl_data[0: int(m*persentage_split)].T  # split data for training
X_train = train_data[0:-1].T                        # get input train data
Y_train = np.expand_dims(train_data[-1], axis=-1)   # get output train data
row, col = train_data.shape

#X_test = 
#Y_test = 

# activation function
def sigmoid(s):                                     # Get Zj and Yk
    return 1 / (1 + np.exp(-s))

# derivative of sigmoid
def sigmoid_derivative(sd):                         # Get dk and dj
    return sd * (1 - sd)

# class definition
class NeuralNetwork:
    def __init__(self, X_train, Y_train, alpha):
        self.input   = X_train                      # input
        self.output  = Y_train                      # output
        #self.alpha   = alpha                       # learning rate
        self.W1      = np.random.rand(row-1, 10)    # 10 nodes in hidden layer
        self.W2      = np.random.rand(10, 1)        # 1 node in output layer 
        
    def feedforward(self):
        self.Z1 = sigmoid(self.input.dot(self.W1))  # input-to-hidden - step 4 
        self.Z2 = sigmoid(self.Z1.dot(self.W2))     # hidden-to-output - step 5
        return self.Z1, self.Z2

    def backprop(self):
        error_rate = (self.output - self.Z2)                   # error in hidden-to-output
        self.dZ1 = error_rate * sigmoid_derivative(self.Z2)  
        self.dZ2_in = self.dZ1.dot(self.W2.T)                  # how much our hidden layer weights contribute to output error
        self.dZ2 = self.dZ2_in * sigmoid_derivative(self.Z1)   # applying derivative of sigmoid to z2 error

    def update_param(self):
        self.W1 += np.dot(self.input.T, self.dZ2)   # updating input-hidden weights
        self.W2 += np.dot(self.Z1.T, self.dZ1)      # updating hidden-output weights
    
    def train(self, X_train, Y_train):
        self.target = self.feedforward()
        self.backprop()
        self.update_param()

def __MAIN__(X_train, Y_train, alpha, iterations):
    NN = NeuralNetwork(X_train, Y_train, alpha)
    for i in range(iterations):
        if (i % 10 == 0):
            Z1, Z2 = NN.feedforward()
            print("Iteration: ", i)
            print("Accuarcy:", np.sum(Z2 == Y_train) / Y_train.size)
            print("Loss: ", str(mean_squared_error(Y_train, Z2)))
            print("\n")
    NN.train(X_train, Y_train)

__MAIN__(X_train, Y_train, 0.1, 100)
