import os
import re
import sys
import wave
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from random import shuffle
import librosa
# visualize training data

#   In the RNN, we do not suggest to use tf.nn.run(cell,inputs) becasue
#   http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
#
#
#
#

import argparse

test_path = "test3/"
path = "data/spoken_numbers_pcm/"

# learning_rate = 0.00001
# training_iters = 300000 #steps
# batch_size = 64

height=20                                       # mfcc features
width=80                                        # (max) length of utterance
classes=10                                      # digits

n_input = 20
n_steps = 80                                    # time steps
n_hidden = 128                                  # number of neurons in the hidden layer
n_classes = 10

# Using GPU to apply AVX2 computation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define learning rate and iteration times
learning_rate = 0.001                           # learning rate
training_iters = 100000                         # training step

# Define batch size
batch_size = 50
display_step = 10

# batch generator
def mfcc_batch_generator(batch_size=10):
    # maybe_download(source, DATA_DIR)
    batch_features = []
    labels = []
    files = os.listdir(path)
    while True:
        # print("loaded batch of %d files" % len(files))
        shuffle(files)
        for file in files:
            if not file.endswith(".wav"): continue
            #print(file)
            wave, sr = librosa.load(path+file, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            label = dense_to_one_hot(int(file[0]),10)
            labels.append(label)
            # print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc,((0,0),(0,80-len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc).T)
            if len(batch_features) >= batch_size:
                yield np.array(batch_features), np.array(labels)
                batch_features = []  # Reset for next batch
                labels = []

def mfcc_batch_generator_for_test(batch_size=10):
    # maybe_download(source, DATA_DIR)
    batch_features = []
    labels = []
    files = os.listdir(test_path)
    while True:
        # print("loaded batch of %d files" % len(files))
        shuffle(files)
        for file in files:
            if not file.endswith(".wav"): continue
            wave, sr = librosa.load(test_path + file, mono=True)
            mfcc = librosa.feature.mfcc(wave, sr)
            label = dense_to_one_hot(int(file[0]), 10)
            labels.append(label)
            # print(np.array(mfcc).shape)
            mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)
            batch_features.append(np.array(mfcc).T)
            if len(batch_features) >= batch_size:
                yield np.array(batch_features), np.array(labels)
                batch_features = []  # Reset for next batch
                labels = []

# visualize training process
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      # calculate the mean value and record it
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)

      # 计算参数的标准差
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      # 用直方图记录参数的分布
      tf.summary.histogram('histogram', var)

def dense_to_one_hot(labels_dense, num_classes=10):
    return np.eye(num_classes)[labels_dense]


# RNN function
def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    act = tf.matmul(outputs[-1], weights['out']) + biases['out']
    tf.summary.histogram("activations", act)
    return act

# main
# active tensor
def main():
    # Setting tensor flow board
    tf.reset_default_graph()
    with tf.name_scope("hidden"):
        weights = {
            # shape (128, 10)
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        biases = {
            # shape (10)
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        tf.summary.histogram("weight", weights['out'])
        tf.summary.histogram("biases", biases['out'])

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, n_steps, n_input], name='x-input')
        y = tf.placeholder(tf.float32, [None, n_classes], name='y-input')

    with tf.name_scope("accuracy"):
        pred = RNN(x, weights, biases)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        tf.summary.scalar("loss", loss)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    # merge all the summary
    summ = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)
        step = 1
        # create the summary writer
        writer = tf.summary.FileWriter("logs", sess.graph)
        while step * batch_size < training_iters:
            batch = mfcc_batch_generator(batch_size)
            batch_x, batch_y = next(batch)
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            if step % display_step == 0:
                [acc, l, s] = sess.run([accuracy, loss, summ], feed_dict={x: batch_x, y: batch_y})
                writer.add_summary(s, step)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss = " + \
                    "{:.6f}".format(l) + ", Training Accuracy = " + \
                    "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")
        total_accuracy = 0
        for i in range(10):
            batch = mfcc_batch_generator_for_test(200)
            batch_x, batch_y = next(batch)
            test_accuracy, s = sess.run([accuracy, summ], feed_dict={x: batch_x, y: batch_y})
            print("Iteration # {}. Test Accuracy: {:.0f}%".format(i + 1, test_accuracy * 100))
            total_accuracy += (test_accuracy / 10)
            writer.add_summary(s, i)
        print(total_accuracy*100)
    # test part

main()