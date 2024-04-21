#Trying to build a neural net that learns the addition operation.

def error(actualOp, targetOp):
    delta = actualOp - targetOp
    return (sum([x**2 for x in delta])/2.0).sum()

def relu(x):
    return max(x,0)

def reluDiff(fx):
    return 1 if fx > 0 else 0

import numpy as np
class NeuralNet:
    def __init__(self,
            inputSize,
            outputSize,
            hiddenLayers):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayers = hiddenLayers
        self.weights = []
        self.biases = []
        self.learningRate = 0.1
        #sigmoid activation fn
        self.activationFn = np.vectorize(lambda x: 1.0/(1+np.exp(-x)))
        self.activationFnDerivative = np.vectorize(lambda fx: fx*(1-fx))


        #ReLU activation fn
        #self.activationFn = np.vectorize(relu)
        #self.activationFnDerivative = np.vectorize(reluDiff)
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


    def test(self, inp):
        self.forwardProp(inp)
        return np.argmax(self.activations[-1])


def genIOWithBounds(inputBound, outputBound):
    a = np.random.randint(0, inputBound)
    b = np.random.randint(0, inputBound)
    while b == a:
        b = np.random.randint(0, inputBound)
    inp = np.zeros((1, inputBound))
    inp[0][a] = 1.0
    inp[0][b] = 1.0

    op = np.zeros((1, outputBound))
    op[0][a+b] = 1.0
    return inp.T, op.T, a, b


def genIO(a, b):
    inp = np.zeros((1, inputBound))
    inp[0][a] = 1.0
    inp[0][b] = 1.0

    op = np.zeros((1, outputBound))
    op[0][a+b] = 1.0
    return inp.T, op.T


if __name__ == '__main__':
    np.random.seed(1)
    inputBound = 7
    outputBound = 2*inputBound
    nn = NeuralNet(inputBound,outputBound,0)
    #print('Initial wts and biases')
    #print(nn.weights, nn.biases)
    #print('\n')

    allInput = [(a,b) for a in range(inputBound) for b in range(a+1,
        inputBound)]
    np.random.shuffle(allInput)
    trainingSize = len(allInput)*7//10
    trainingInput = allInput[:trainingSize]
    testInput = allInput[trainingSize:]


    ix = 0
    for _ in range(42000):
        a, b = trainingInput[ix]
        inp, op = genIO(a, b)
        nn.iterate(inp, op)
        ix = (ix + 1) % len(trainingInput)

    mismatches = 0

    for (a,b) in testInput:
        inp, op = genIO(a, b)
        result = nn.test(inp)
        if a+b != result:
            print(a, b, result)
            print(nn.activations[-1])
            mismatches += 1

    print(f'Test input size = {len(testInput)}')
    print(f'Total addition mismatches = {mismatches}') 

