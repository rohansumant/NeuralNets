# https://github.com/miloharper/simple-neural-network/blob/master/short_version.py

from numpy import exp, array, random, dot
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1)) - 1
print(synaptic_weights)
for iteration in range(2):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    xx = dot(training_set_inputs.T, (training_set_outputs - output) * output
            * (1 - output))
    print('Wt updates: ',xx)
    synaptic_weights += xx
print (1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))
