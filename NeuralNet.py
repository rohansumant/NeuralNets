#Trying to build a neural net that learns the addition operation.

def error(actualOp, targetOp):
    delta = actualOp - targetOp
    return (sum([x**2 for x in delta])/2.0).sum()

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
        self.learningRate = 0.1
        #sigmoid activation fn
        self.activationFn = np.vectorize(lambda x: 1.0/(1+np.exp(-x)))
        self.activationFnDerivative = np.vectorize(lambda fx: fx*(1-fx))
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
            currActivation = self.activationFn(currOp + self.biases[i])
            activations.append(currActivation)
        self.activations = activations
        #print(f'activations: {self.activations}\n')

    def backProp(self, actualOp, targetOp):
        delta = actualOp - targetOp
        #print(delta)
        activations = self.activations
        wPtr = len(self.weights)-1
        aPtr = len(activations)-1
        # calculate delta for each layer except input
        D = []
        for i in range(self.hiddenLayers + 1, 0, -1):
            if len(D) == 0:
                D.append(delta * self.activationFnDerivative(activations[aPtr]))
            else:
                prevDelta = D[-1]
                currDelta = (self.weights[wPtr].dot(prevDelta)
                        * self.activationFnDerivative(activations[aPtr]))
                D.append(currDelta)
                wPtr -= 1
            aPtr -= 1

        D = D[::-1]

        #print(f'Delta {D}\n')
        for i in range(self.hiddenLayers + 1):
            wtUpdate = self.learningRate * (activations[i].dot(D[i].T))
            biasUpdate = self.learningRate * (D[i])
            self.weights[i] -= wtUpdate
            self.biases[i] -= biasUpdate


    def iterate(self, inp, targetOp):
        self.forwardProp(inp)
        actualOp = self.activations[-1]
        err = error(actualOp, targetOp)
        print(f'Error: {err}')
        self.backProp(actualOp, targetOp)



if __name__ == '__main__':
    nn = NeuralNet(2,2,1)
    print('Initial wts and biases')
    print(nn.weights, nn.biases)
    print('\n')
    inp = np.array([0,1]).reshape(2,1)
    op = np.array([1,0]).reshape(2,1)

    for _ in range(1000):
        nn.iterate(inp, op)

    print(nn.activations)
