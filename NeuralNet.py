'''
Trying to build a neural net that learns the addition operation.

The input vector will be [1..n] and output vector should be [3..(2*n-1)]
'''

import numpy as np

'''
Creating a feed forward neural net with default 1 hidden layer.
'''
def create_neural_net(input_size, hidden_layer_count=1):
    # output for input_size = n will be (2*n-3)
    hidden_layer_size = output_size = 2*input_size
    weights = []
    biases = []
    for i in range(hidden_layer_count+1):
        rows, cols = -1, -1
        if i == 0:
            # first hidden layer
            rows = input_size
        else:
            rows = hidden_layer_size
        cols = hidden_layer_size
        assert(min(rows,cols) > 0)

        weight_matrix = np.random.rand(rows,cols)
        weights.append(weight_matrix)

        biases.append(np.random.rand(1,hidden_layer_size))

    return weights, biases

def activation_fn(x):
    # ReLU
    return max(x,0)

def apply_input(net, input_vector: np.ndarray):
    (weights, biases) = net
    layer_count = len(weights) # not including the input layer here
    assert(layer_count == len(biases))

    curr_inp = np.copy(input_vector)
    accum = []
    for i in range(layer_count):
        wt_matrix = weights[i]
        bias = biases[i]

        mm_result = np.matmul(curr_inp, wt_matrix)
        assert(mm_result.shape == bias.shape)
        assert(mm_result.shape[0] == 1)
        net = mm_result + bias
        activated_op = np.vectorize(activation_fn)(net)
        #accum.append((net, activated_op))
        accum.append(activated_op)
        curr_inp = np.copy(activated_op)

    return accum


def error(v1, v2):
    assert(v1.shape == v2.shape and v1.shape[0] == 1)
    result = np.vectorize(lambda x:x**2)(v1-v2)
    result /= 2.0
    return result


def main():
    net = create_neural_net(3)
    input_vector = np.array([[0,1,1]], float)
    
    op = apply_input(net, input_vector)[-1]
    expected_op = np.zeros((1,6))
    expected_op[0][3] = 1.0

    err = error(op, expected_op)
    print(f'{err} {np.linalg.norm(err)}')



main()
