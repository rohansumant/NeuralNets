'''
Trying to build a neural net that learns the addition operation.

The input vector will be [1..n] and output vector should be [3..(2*n-1)].
For simplicity, output vector size is maintained to be [1..2*n].
'''

import numpy as np

'''
Creating a feed forward neural net with default 1 hidden layer.
'''

learning_rate = 1e-2

def apply_fn(np_array, fn):
    return np.vectorize(fn)(np_array)

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

        #biases.append(np.random.rand(1,hidden_layer_size))
        biases.append(np.zeros((1,hidden_layer_size)))

    return weights, biases

def activation_fn(x):
    # ReLU
    return max(x,0)

def activation_fn_diff(y):
    # ReLU diff
    # Input value y = ReLU(x)
    if y == 0:
        return 0
    else:
        return 1

def apply_input(net, input_vector: np.ndarray):
    (weights, biases) = net
    layer_count = len(weights) # not including the input layer here
    assert(layer_count == len(biases))

    accum = [np.copy(input_vector)]
    for i in range(layer_count):
        curr_inp = accum[-1]
        wt_matrix = weights[i]
        bias = biases[i]
        mm_result = np.matmul(curr_inp, wt_matrix)
        assert(mm_result.shape == bias.shape)
        assert(mm_result.shape[0] == 1)
        activated_op = apply_fn(mm_result + bias, activation_fn)
        #accum.append((net, activated_op))
        accum.append(activated_op)

    return accum


def error(v1, v2):
    assert(v1.shape == v2.shape and v1.shape[0] == 1)
    result = np.vectorize(lambda x:x**2)(v1-v2)
    result /= 2.0
    return result


def backpropagate(net, activations, op_delta):
    (wts, biases) = net
    wts_len = len(wts)
    assert(wts_len == len(activations)-1)
    wt_deltas = [None] * wts_len
    wt_deltas[-1] = np.copy(op_delta) * (apply_fn(activations[-1],activation_fn_diff))

    #iterate on wts_len-1, because 1 entry is already populated above
    for i in range(wts_len-1, 0, -1):
        curr_wt_mat = wts[i]
        delta_mat = np.copy(curr_wt_mat)
        prev_delta = wt_deltas[i]
        assert(prev_delta.shape[0] == 1)
        assert(delta_mat.shape[1] == prev_delta.shape[1])
        # multiplying column wise
        delta_mat *= prev_delta
        # collapse columns together | i.e. sum up rows
        delta_mat = delta_mat.sum(axis=1)
        cols = delta_mat.shape[0]
        delta_mat = delta_mat.reshape((1, cols))
        # derivative of the activation that's input to the current wt. matrix
        activation_delta = apply_fn(activations[i], activation_fn_diff)
        #print(activation_delta.shape, delta_mat.shape)
        assert(activation_delta.shape == delta_mat.shape)
        assert(i-1 >= 0)
        wt_deltas[i-1] = delta_mat * activation_delta


    assert(type(wt_deltas[0]) != type(None))
    print(f'wt_deltas: {wt_deltas}')

    updated_wts = []
    for i in range(wts_len):
        updated_wt_mat = np.copy(wts[i])
        rows, cols = wts[i].shape
        input_activations = activations[i]
        delta_from_layers_ahead = wt_deltas[i]
        for r in range(rows):
            for c in range(cols):
                curr_delta = input_activations[0][r]*delta_from_layers_ahead[0][c]
                updated_wt_mat[r][c] -= learning_rate*curr_delta
        updated_wts.append(updated_wt_mat)

    return updated_wts


def gen_io_vectors(inp_size, op_size):
    input_vector = np.zeros((1,inp_size), float)
    a = np.random.randint(0,inp_size)
    b = np.random.randint(0,inp_size)
    while b == a:
        b = np.random.randint(0,inp_size)
    c = a+b
    assert(c < op_size)
    input_vector[0][a] = 1.0
    input_vector[0][b] = 1.0

    output_vector = np.zeros((1,op_size), float)
    output_vector[0][c] = 1.0

    return input_vector, output_vector


def training_instance(net, input_vector, expected_op):
    biases = net[1]

    #input_vector = np.array([[0,1,1]], float)
    #expected_op = np.zeros((2,6))
    #expected_op[0][3] = 1.0

    #input_vector, expected_op = gen_io_vectors(3,6)
    #print(input_vector, expected_op)


    activations = apply_input(net, input_vector)
    curr_err = np.sum(error(activations[-1], expected_op))

    iteration_cnt = 1000
    while iteration_cnt > 0 and curr_err > 0.001:
        #activations = apply_input(net, input_vector)
        #curr_err = np.sum(error(activations[-1], expected_op))

        delta = activations[-1] - expected_op
        updated_wts = backpropagate(net,
                activations, delta)

        activations_after_backprop = apply_input((updated_wts, biases), input_vector)
        error_after_backprop = np.sum(error(activations_after_backprop[-1],
            expected_op))

        if error_after_backprop >= curr_err:
            # learning rate is too high
            print(f'Error increasing after backprop {error_after_backprop}')
            print(f'debug1: {activations[-1]} {delta}')

            #global learning_rate
            #learning_rate *= 0.25
            #if learning_rate < 1e-8:
                #break
            #iteration_cnt -= 1
            #continue
            break

        print(f'Error: {error_after_backprop} {curr_err}')
        curr_err = error_after_backprop
        #if(curr_err == 0.5 and np.sum(activations[-1]) == 0):
            #print(activations, updated_wts)
            #break
        activations = activations_after_backprop
        net = (updated_wts, biases)
        iteration_cnt -= 1

    print(f'Finished training instance: Error {curr_err} {activations[-1]}')
    return net




def main():

    net = create_neural_net(3)

    for sample in range(1):
        input_vector, expected_op = gen_io_vectors(3,6)
        net = training_instance(net, input_vector, expected_op)

    print(f'Final wts: {net[0]}')


main()
