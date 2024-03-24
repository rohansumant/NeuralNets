'''
Trying to build a neural net that learns the addition operation.

The input vector will be [1..n] and output vector should be [3..(2*n-1)]
'''

import numpy as np

'''
Creating a feed forward neural net with default 1 hidden layer.
'''

learning_rate = 0.01

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

    curr_inp = np.copy(input_vector)
    accum = [np.copy(input_vector)]
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



def update_weights_before_output(wts, op, ip, delta):
    (rows, cols) = wts.shape
    result = np.copy(wts)
    for r in range(rows):
        for c in range(cols):
            wt_delta = delta[0][c]*activation_fn_diff(op[0][c])*ip[0][r]
            result[r][c] -= learning_rate*wt_delta

    #print('op layer wt change')
    #print(wts, result)
    return result

def update_weights(wts, wts_ahead, updated_wts_ahead,
        op, ip):
    (rows, cols) = wts.shape
    result = np.copy(wts)

    for r in range(rows):
        for c in range(cols):
            if op[0][c] == 0:
                continue
            base = (ip[0][r]*activation_fn_diff(op[0][c]))/op[0][c]
            sigma = np.sum(wts_ahead[c].dot(updated_wts_ahead[c]))
            wt_delta = base*sigma
            result[r][c] -= learning_rate*wt_delta

    #print('hidden layer wt change')
    #print(wts, result)
    return result


def backpropagate(net, activations, delta):
    (wts, biases) = net
    updated_wts = [np.copy(i) for i in wts]
    updated_wts[-1] = update_weights_before_output(wts[-1],
            activations[-1],
            activations[-2],
            delta)

    n = len(wts)
    assert(len(activations) == n+1)
    for i in range(n-2,-1,-1):
        updated_wts[i] = update_weights(wts[i], wts[i+1],
                updated_wts[i+1],
                activations[i+1],
                activations[i])

    return updated_wts


def main():
    net = create_neural_net(3)
    biases = net[1]

    input_vector = np.array([[0,1,1]], float)
    expected_op = np.zeros((1,6))
    expected_op[0][3] = 1.0

    err_accum = []

    activations = apply_input(net, input_vector)
    curr_err = np.sum(error(activations[-1], expected_op))

    iteration_cnt = 100
    while iteration_cnt > 0 and curr_err > 0.01:
        #activations = apply_input(net, input_vector)
        #curr_err = np.sum(error(activations[-1], expected_op))

        delta = activations[-1] - expected_op
        updated_wts = backpropagate(net,
                activations, delta)

        activations_after_backprop = apply_input((updated_wts, biases), input_vector)
        error_after_backprop = np.sum(error(activations_after_backprop[-1],
            expected_op))

        if error_after_backprop > curr_err:
            # learning rate is too high
            print('Error increasing after backprop. Skipping net update and \
                    reducing learning rate')
            global learning_rate
            learning_rate *= 0.1
            if learning_rate < 1e-6:
                break
            iteration_cnt -= 1
            continue

        print(f'Error: {error_after_backprop} {curr_err}')
        curr_err = error_after_backprop
        activations = activations_after_backprop
        net = (updated_wts, biases)
        iteration_cnt -= 1

    print(f'Error {curr_err} {activations[-1]}')


main()
