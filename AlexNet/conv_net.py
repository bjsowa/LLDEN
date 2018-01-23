import tensorflow as tf
import numpy as np

# 2D convolution function
def conv2d(x, W, b, strides=1, BN_bool=False):
    # tf.nn.conv2d(input, filter/weights, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    # input: A 4-D Tensor with shape [batch, height, width, channels]. Must be one of the following types: half, float32, float64.
    # filter: A Tensor. Must have the same type as input.
    # strides: A list of ints. 1-D of length 4. The stride of the sliding window for each dimension of input. Must be in the same order as the dimension specified with format.
    # padding: A string from: "SAME", "VALID". The type of padding algorithm to use.
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    if BN_bool:
        x = tf.contrib.layers.batch_norm(x)
    else:
        x = tf.nn.bias_add(x, b)
    x = tf.nn.relu(x)
    return x

# Max pooling function
def maxpool2d(x, z=3, k=2):
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    # value: A 4-D Tensor with shape [batch, height, width, channels] and type tf.float32.
    # ksize: A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
    # strides: A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor.
    # padding: A string, either 'VALID' or 'SAME'. The padding algorithm. See the comment here
    return tf.nn.max_pool(x, ksize=[1, z, z, 1], strides=[1, k, k, 1],
                          padding='SAME')


# The conv net model
def alex_net(x, n_classes, dropout, BN_bool):
    # Store layers weight & bias
    weights = {
        # Input image 224x224x3, 96 kernels of size 11x11x3
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
        # Input activation map 55x55x96, 256 kernels of size 5x5x48
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
        # Input activation map 27x27x256, 384 kernels of size 3x3x256
        'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
        # Input activation map 13x13x384, 384 kernels of size 3x3x384
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        # Input activation map 13x13x384, 256 kernels of size 3x3x384
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
        # Input activation map 7x7x256, 4096 outputs
        'wd1': tf.Variable(tf.random_normal([7*7*256, 4096])),
        # Input activation map 4096 vector, 4096 outputs
        'wd2': tf.Variable(tf.random_normal([4096, 4096])),
        # 4096 inputs, 20 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([4096, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([4096])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # Reshape input to be a 2D array
    # Why reshape to -1 is a placeholder that says "adjust as necessary to match the size needed for the full tensor." 
    # It's a way of making the code be independent of the input batch size, 
    # so that you can change your pipeline and not have to adjust the batch size everywhere in the code.
    x = tf.reshape(x, shape=[-1, 224, 224, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], strides=4, BN_bool=BN_bool)
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, z=3, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], strides=1, BN_bool=BN_bool)
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, z=3, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], strides=1, BN_bool=BN_bool)
    # Convolution Layer
    conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], strides=1, BN_bool=BN_bool)
    # Convolution Layer
    conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], strides=1, BN_bool=BN_bool)
    conv5 = maxpool2d(conv5, z=3, k=2)

    # Fully connected layer
    # Reshape conv5 output to fit the FC layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    if BN_bool:
        fc1 = tf.matmul(fc1, weights['wd1'])
        fc1 = tf.contrib.layers.batch_norm(fc1)
    else:
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    if BN_bool:
        fc2 = tf.matmul(fc1, weights['wd2'])
        fc2 = tf.contrib.layers.batch_norm(fc2)
    else:
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)

    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    
    return out