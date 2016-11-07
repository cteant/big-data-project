from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import cPickle as pickle

import argparse
import tensorflow as tf
import numpy as np
FLAGS = None


def main(_):
  #read data
  f = open( "notMNIST.pickle",'rb')
  data = pickle.load(f)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  # Train
  tf.initialize_all_variables().run()
  batch_size = 100
  for i in range(4500):
    batch_xs = data['train_dataset'][batch_size*i:batch_size*(i+1),:,:].reshape(batch_size,-1)
    batch_ys = data['train_labels'][batch_size*i:batch_size*(i+1)]
    batch_ys_new = np.zeros((batch_size,10),dtype=np.int8)
    for i in range(batch_size):
        batch_ys_new[i,batch_ys[i]]=1

    if i%100 == 0:
       train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys_new})
       print("Step %d, accuracy %g"% (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys_new})

  # Test trained model
  #print(mnist.train.images[0])
  #print(mnist.train.labels[0])
  batch_xt = data['test_dataset'].reshape(data['test_dataset'].shape[0],-1)
  batch_yt = data['test_labels']
  batch_yt_new = np.zeros((data['test_labels'].shape[0],10),dtype=np.int8)
  for i in range(data['test_labels'].shape[0]):
      batch_yt_new[i,batch_yt[i]]=1

  print(sess.run(accuracy, feed_dict={x: batch_xt,
                                      y_: batch_yt_new}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  print(FLAGS)
  tf.app.run()
