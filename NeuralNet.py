#Trying to build a neural net that learns the addition operation.
import numpy as np
class NeuralNet:
    def __init__(self,
            inputSize,
            outputSize,
            hiddenLayers):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayers = hiddenLayers
        np.random.seed(1)
        self.weights = []
        self.biases = []
        self.relu = np.vectorize(lambda x: max(0,x))
        self.reluDerivative = np.vectorize(lambda fx: 0 if fx < 0 else 1)
        for i in range(self.hiddenLayers + 1):
            if i == 0:
                self.weights.append(np.random.random((inputSize, outputSize)))
            else:
                self.weights.append(np.random.random((outputSize, outputSize)))
            self.biases.append(np.random.random((outputSize, 1)))
    def forwardProp(self, inp):
        activations = [inp]
        for i in range(self.hiddenLayers + 1):
            currInp = activations[-1]
            currOp = (currInp.T.dot(self.weights[i])).T
            currActivation = self.relu(currOp + self.biases[i])
            activations.append(currActivation)
        self.activations = activations
    def backProp(self, actualOp, targetOp):
        delta = actualOp - targetOp
        activations = self.activations
        wPtr = len(self.weights)-1
        # calculate delta for each layer except input
        D = []
        for i in range(self.hiddenLayers + 1, 0, -1):
            prevDelta = None if len(D) == 0 else D[-1]
            if not prevDelta:
                D.append(delta * reluDerivative(activations[-1]))
            else:
                D.append(self.weights[wPtr].dot(prevDelta))
                wPtr -= 1
        D = D[::-1]
        for i in range(self.hiddenLayers + 1):
            update = self.learningRate * (activations[i].T.dot(activations[i]))
            self.weights[i] -= update
    def error(self, actualOp, targetOp):
        delta = actualOp - targetOp
        return sum([x**2 for x in delta])/2.0



if __name__ == '__main__':
    nn = NeuralNet(3,6,1)
    print(nn.weights, nn.biases)
    inp = np.array([0,0,1]).reshape(3,1)
    nn.forwardProp(inp)
    print(nn.weights, nn.biases)
