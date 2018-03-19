#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from helper import *
import time

class SRCNN(object):

    def __init__(
        self,
        sess,
        train_data_in,
        train_data_out,
        test_data_in,
        test_data_out,
        learning_rate=0.01,
        image_size_x=475,
        image_size_y=476,
        num_epochs=1000,
        ):

        self.sess = sess
        self.train_data_in = train_data_in
        self.train_data_out = train_data_out
        self.test_data_in = test_data_in
        self.test_data_out = test_data_out

        self.learning_rate = learning_rate
        (self.image_size_x, self.image_size_y) = (image_size_x,
                image_size_y)
        self.num_epochs = num_epochs
        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [1,self.image_size_x, self.image_size_y,1], name='inputs')
        self.outputs = tf.placeholder(tf.float32, [1,self.image_size_x, self.image_size_y,1], name='outputs')

        self.weights = {'w1': tf.Variable(tf.random_normal([9, 9, 1,
                        64], stddev=1e-3), name='w1'),
                        'w2': tf.Variable(tf.random_normal([1, 1, 64,
                        32], stddev=1e-3), name='w2'),
                        'w3': tf.Variable(tf.random_normal([5, 5, 32,
                        1], stddev=1e-3), name='w3')}

        self.biases = {'b1': tf.Variable(tf.zeros([64]), name='b1'),
                       'b2': tf.Variable(tf.zeros([32]), name='b2'),
                       'b3': tf.Variable(tf.zeros([1]), name='b3')}

        self.pred = self.model()

        # Loss function (MSE)

        self.loss = tf.reduce_mean(tf.square(self.outputs - self.pred))

    def train(self):
        self.train_op = \
            tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()

        counter = 0
        start_time = time.time()

        print('Training....')

        for ep in range(self.num_epochs):
            total_images = train_data_in.shape[-1]
            sx,sy = train_data_in[:,:,0].shape
            for i in range(total_images):
                inputs = np.reshape(train_data_in[:, :, i],(1,sx,sy,1))
                outputs = np.reshape(train_data_out[:, :, i],(1,sx,sy,1))
            counter += 1
            (_, err) = self.sess.run([self.train_op, self.loss],
                    feed_dict={self.inputs: inputs,
                    self.outputs: outputs})
            if counter % 10 == 0:
                print('Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]' \
                    % (ep + 1, counter, time.time() - start_time, err))

    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.inputs, self.weights['w1'
                           ], strides=[1, 1, 1, 1], padding='SAME')
                           + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'],
                           strides=[1, 1, 1, 1], padding='SAME')
                           + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1,
                             1, 1], padding='SAME') + self.biases['b3']

        return conv3

    def test(self):
        print('Testing....')
        total_images = test_data_in.shape[-1]
        sx,sy = test_data_in[:,:,0].shape
        for i in range(total_images):
            x = np.reshape(test_data_in[:,:,i],(1,sx,sy,1))
            y = np.reshape(test_data_out[:,:,i],(1,sx,sy,1))
            #result = self.pred.eval({self.inputs: x,self.outputs: y})
            y_hat = self.pred.eval({self.inputs:x})
        print(compare_psnr(y.astype(np.float32) , y_hat))
