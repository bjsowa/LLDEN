from __future__ import print_function

import os
import time, datetime
import sys

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.core.framework import types_pb2
import numpy as np
import conv_net
import load_pascal_voc

pascal_voc_path = os.path.abspath("VOC2012")
labels_subdir = pascal_voc_path + "/Classification_Training"
images_subdir = pascal_voc_path + "/JPEGImages"
valid_labels_subdir = pascal_voc_path + "/Classification_Validation"

def LOG(X, f=None):
    time_stamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    if not f:
        print(time_stamp + " " + X)
    else:
        f.write(time_stamp + " " + X)


# Retrieve the training data        
training_data = load_pascal_voc.load_pascal_voc_classification(pascal_voc_path, labels_subdir, images_subdir)
# print(training_data)
validation_data = load_pascal_voc.load_pascal_voc_classification(pascal_voc_path, valid_labels_subdir, images_subdir)
# print(validation_data)

# Parameters
learning_rate = 0.005
training_iters = 300000
batch_size = 64
display_step = 20
size = 224
n_classes = 20 
dropout = 0.5 # Dropout, probability to keep units
BN_bool = True # Optional Batch Normalization

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label

# Reads paths of images together with their labels
image_list = [images_subdir+"/"+x[0]+".jpg" for x in training_data]
label_list = [x[1] for x in training_data]
valid_image_list = [images_subdir+"/"+x[0]+".jpg" for x in validation_data]
valid_label_list = [x[1] for x in validation_data]

images = ops.convert_to_tensor(image_list, tf.string)
labels = ops.convert_to_tensor(label_list, tf.float32)
valid_images = ops.convert_to_tensor(image_list, tf.string)
valid_labels = ops.convert_to_tensor(label_list, tf.float32)

# Makes an input queue
input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
valid_input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

image, label = read_images_from_disk(input_queue)
valid_image, valid_label = read_images_from_disk(valid_input_queue)

def preprocess_image(img, size, colour):
    img = tf.cast(img, tf.float32)
    img = tf.image.resize_images(img, [size, size])
    if not colour:
        img = tf.image.rgb_to_grayscale(img)

    # img = tf.image.per_image_standardization(img)
    return img

colour = True
image = preprocess_image(valid_image, 224, colour)
valid_image = preprocess_image(valid_image, 224, colour)
channels = 3
if not colour:
    channels = 1

# Optional Image and Label Batching
image_batch, label_batch = tf.train.batch([image, label],
                                          batch_size=batch_size)
valid_image_batch, valid_label_batch = tf.train.batch([valid_image, valid_label],
                                          batch_size=batch_size)

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

  step = 1

  # Keep training until we reach the max iterations
  while step * batch_size < training_iters:
      
    # Run the optimization 
    batch_x, batch_y = sess.run([image_batch, label_batch])

    sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
    if step % display_step == 0:
          # Calculate batch loss and accuracy
          loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})

          LOG("Iteration # " + str(step*batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
    step += 1
    # print(sess.run(label_batch))
  LOG("Optimization Finished!")

  # Calculate accuracy 
  valid_batch_x, valid_batch_y = sess.run([valid_image_batch, valid_label_batch])
  print("Testing Accuracy:", \
      sess.run(accuracy, feed_dict={x: valid_batch_x,
                                    y: valid_batch_y,
                                    keep_prob: 1.}))

  coord.request_stop()
  coord.join(threads)