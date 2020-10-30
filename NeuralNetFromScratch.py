########################################################################################################################
# Neural Network without Hidden Layers (Perceptron) made by Aniket N Prabhu ############################################
# Reference: https://www.youtube.com/watch?v=kft1AJ9WVDk ###############################################################
########################################################################################################################

import numpy as np

# Step 1: Defining the normalizing function


def sigmoid(x):  # Normalizing sigmoid function
    y = 1 / (1 + np.exp(-x))
    return y


def sigder(x):  # sigmoid derivative function
    f = x * (1 - x)
    return f


# Step 2: Defining the training data
tip = np.array([[0, 0, 1],  # Training inputs. "np.array" is used to create a numpy array/matrix instead of a list.
                [1, 1, 1],
                [1, 0, 1],
                [0, 1, 1]])

top = np.array([[0],
                [1],
                [1],
                [0]])

# It can also be written as "top = np.array([[0, 1, 1, 0]]).T", where "T" indicates "transpose".

# Step 3: assigning random values to weights
np.random.seed(1)

# We will now create a 3x1 matrix of synaptic weights because we have 3 inputs corresponding to 1 output
syn_wts = 2 * np.random.random((3, 1)) - 1
print("Random starting synaptic weights: \n", syn_wts)

# Without hidden layers, the flow is just: input_layer(ipl) -> neuron -> output_layer(opl)
for i in range(20000):  # Training the model 20000 times. More = better but only till a certain point.
    ipl = tip  # Input Layer
    opl = sigmoid(np.dot(ipl, syn_wts))  # Output Layer
    err = top - opl  # Error
    waf = err * sigder(opl)  # Weight Adjustment Factor
    syn_wts += np.dot(ipl.T, waf)  # Updated Synaptic Weights
print("Post training synaptic weights:\n", syn_wts)
print("Post training outputs: \n", opl)


