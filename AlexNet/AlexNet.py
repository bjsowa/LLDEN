from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import datetime

import numpy as np
from sklearn import datasets
import tensorflow as tf
import torchvision

import conv_net

# Load Cifar-100 dataset

data_path = os.environ.get('PYTORCH_DATA_PATH', 'data')

cifar_full_train_dataset = torchvision.datasets.CIFAR100(
    data_path, train=True, download=True)

cifar_test_dataset = torchvision.datasets.CIFAR100(
    data_path, train=False, download=True)

cifar_full_train_data = np.array(cifar_full_train_dataset.train_data)
cifar_full_train_labels = np.array(cifar_full_train_dataset.train_labels)

cifar_test_data = np.array(cifar_test_dataset.test_data)
cifar_test_labels = np.array(cifar_test_dataset.test_labels)
cifar_train_data = cifar_full_train_data[:40000]
cifar_train_labels = cifar_full_train_labels[:40000]
cifar_valid_data = cifar_full_train_data[40000:]
cifar_valid_labels = cifar_full_train_labels[40000:]

# Parameters
epochs = 10
batch_size = 10
display_step = 20

learning_rate = 0.005
size = 32
channels = 3
n_classes = 100
dropout = 0.5 # Dropout, probability to keep units
BN_bool = True # Optional Batch Normalization

train_images = tf.convert_to_tensor(cifar_train_data, tf.float32)
train_labels = tf.convert_to_tensor(cifar_train_labels, tf.float32)
valid_images = tf.convert_to_tensor(cifar_valid_data, tf.float32)
valid_labels = tf.convert_to_tensor(cifar_valid_labels, tf.float32)


def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# tf Graph input
x = tf.placeholder(tf.float32, [None, size, size, channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # Dropout (keep probability)

# Construct model
pred = conv_net.alex_net(x, n_classes, keep_prob, BN_bool)
pred = tf.reshape(pred, [-1, n_classes])

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
  
    # Initialize the variables
    sess.run(tf.global_variables_initializer())

    # Start the batch queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    LOG("from the train set:")

    for epoch_no in range(epochs):
        for index, offset in enumerate(range(0, 40000, batch_size)):
            batch_x = cifar_train_data[offset: offset + batch_size]

            batch_labels = cifar_train_labels[offset: offset + batch_size]
            batch_y = np.zeros((batch_size, n_classes))
            batch_y[np.arange(batch_size), batch_labels] = 1

            #print( batch_y )

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                keep_prob: dropout})


            if index % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                      y: batch_y,
                                                      keep_prob: 1.})

                LOG("Iteration # " + str(index) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc))

  #   step += 1
  #   # print(sess.run(label_batch))
  # LOG("Optimization Finished!")

  # # Calculate accuracy 
  # valid_batch_x, valid_batch_y = sess.run([valid_image_batch, valid_label_batch])
  # print("Testing Accuracy:", \
  #     sess.run(accuracy, feed_dict={x: valid_batch_x,
  #                                   y: valid_batch_y,
  #                                   keep_prob: 1.}))

  # coord.request_stop()
  # coord.join(threads)