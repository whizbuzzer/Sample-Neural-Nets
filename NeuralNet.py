########################################################################################################################
# A more usable perceptron using "class" by Aniket N Prabhu ############################################################
# Reference: https://www.youtube.com/watch?v=Py4xvZx-A1E    ############################################################
########################################################################################################################

import numpy as np

# Step 1: Defining a class. Class consists of a bunch of functions.

class NeuralNet():  # Class name is always written in camel case i.e. beginning/middle letters would be capital and
                    # there would be no space between words.

    def __init__(self):  # Initialization function. Class variables are defined within this function.
                         # Putting 2 underscores(_) before and after function name is a naming convention
                         # called Dunder/magic method or "Data model method". "self" is a "standard variable" and a
                         # namespace taken by every function inside of "class". Other variables can be solved inside
                         # "self" which can later be used outside and inside the neural network.
        np.random.seed(1)
        # Now we will use "self" namespace to store our synaptic weights
        self.syn_wts = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigder(self, x):
        return x * (1 - x)

    # Next we will define "train" and "think" functions which are the core of our object
    # "train" and "think" are arbitrarily chosen names.

    def train(self, trip, trop, triter):  # trip = training inputs & trop = training outputs.
        for i in range(triter):
            op = self.think(trip)  # output
            err = trop - op        # error
            waf = np.dot(trip.T, err * self.sigder(op))
            self.syn_wts += waf    # synaptic weights needed for ack propagation

    def think(self, ip):
        # inputs are integers but the synaptic weights are floats. Hence, we would first need to convert
        # our inputs to floats
        ip = ip.astype(float)
        op = self.sigmoid(np.dot(ip, self.syn_wts))
        return op

# This concludes the class
# Step 2: We will make a usable commandline program:
if __name__ == "__main__":
    # Now we will initialize a neural network:
    neural_net = NeuralNet()
    print("\nRandom Synaptic Weights:\n", neural_net.syn_wts)  # Calling the "syn_wts" method on the "NeuralNetwork"
                                                               # class
    tip = np.array([[0, 0, 1],  # Training inputs. "np.array" is used to create a numpy array/matrix instead of a list.
                    [1, 1, 1],
                    [1, 0, 1],
                    [0, 1, 1]])

    top = np.array([[0],
                    [1],
                    [1],
                    [0]])

    titer = 100000  # training iterations = 100000

    neural_net.train(tip, top, titer)

    print("\nSynaptic Weights post training:\n", neural_net.syn_wts)
    # Now we will ask the user for custom inputs
    A = str(input("\nInput 1:"))
    B = str(input("\nInput 2:"))
    C = str(input("\nInput 3:"))
    print("\nNew inputs:", A, B, C)
    print("\nOutputs:", neural_net.think(np.array([A, B, C])))




