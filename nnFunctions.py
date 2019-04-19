import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() and preprocess()
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def preprocess(filename,scale=True):
    '''
     Input:
     filename: pickle file containing the data_size
     scale: scale data to [0,1] (default = True)
     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    '''
    with open(filename, 'rb') as f:
        train_data = pickle.load(f)
        train_label = pickle.load(f)
        test_data = pickle.load(f)
        test_label = pickle.load(f)
    # convert data to double
    train_data = train_data.astype(float)
    test_data = test_data.astype(float)

    # scale data to [0,1]
    if scale:
        train_data = train_data/255
        test_data = test_data/255

    return train_data, train_label, test_data, test_label

def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # your code here - remove the next four lines
    neg_z = -1.0 * z
    s = 1.0 / (1.0 + np.exp(neg_z))

    return s


def feedForwardPass(train_data, W1, W2):

    bias_a = np.ones((len(train_data), 1))
    X_ip = np.concatenate((train_data, bias_a), axis=1)
    a = np.dot(X_ip, np.transpose(W1))
    hidden = sigmoid(a)

    bias_b = np.ones((len(hidden), 1))
    Z_ij = np.concatenate((hidden, bias_b), axis=1)
    b = np.dot(Z_ij, np.transpose(W2))
    output = sigmoid(b)

    return (output, X_ip, Z_ij)

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not including the bias node)
    % n_hidden: number of node in hidden layer (not including the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % train_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0


    tuple = feedForwardPass(train_data, W1, W2)
    output = tuple[0]
    X_ip = tuple[1]
    Z_ij = tuple[2]


    # 10fk
    y = np.zeros((len(train_label), n_class))
    for row in range(len(train_label)):
        col = train_label[row]
        y[row][col] = 1

    # eq5
    n = len(train_data)
    inner_sum = np.sum((y*np.log(output)) + ((1-y) * np.log(1-output)))
    error = (-1/n) * inner_sum

    # eq15 regularization
    regularization = (lambdaval / (2*n)) * (np.sum(W1 * W1) + np.sum(W2 * W2))
    error_regularization = error + regularization
    obj_val = error_regularization

    # eq9
    delta = output - y
    derivative_W2 = np.dot(np.transpose(delta), Z_ij)

    # eq 16
    lambda_W2 = lambdaval * W2
    sum_W2_lambda = derivative_W2 + lambda_W2
    grad_W2 = (1/n) * sum_W2_lambda

    # eq 12
    delta_W2_dot = np.dot(delta, W2)
    derivative_W1 = np.dot( ( np.transpose((1 - Z_ij) * Z_ij * delta_W2_dot) ), X_ip)

    # eq 17
    lambda_W1 = lambdaval * W1
    # print(lambda_W1.shape)
    # print(derivative_W1.shape)

    # remove row from matrix derivative_W1
    derivative_W1_fixed = np.delete(derivative_W1, derivative_W1.shape[0] - 1, axis=0)

    sum_W1_lambda = derivative_W1_fixed + lambda_W1


    # print(sum_W1_lambda.shape)

    grad_W1 = (1/n) * sum_W1_lambda

    # Make sure you reshape the gradient matrices to a 1D array. for instance if
    # your gradient matrices are grad_W1 and grad_W2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_W1.flatten(), grad_W2.flatten()),0)
    # obj_grad = np.zeros(params.shape)

    return (obj_val, obj_grad)



def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels
    '''

    labels = np.zeros((data.shape[0],))
    # Your code here

    tuple = feedForwardPass(data, W1, W2)
    output = tuple[0]
    labels = np.argmax(output, axis=1)


    return labels
