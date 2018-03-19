import os
import pprint
import numpy as np
import tensorflow as tf
from helper import *
from model import *

pp = pprint.PrettyPrinter()

def main(_):
    observed = read_from_mat('stack.mat')['stack']
    ground  = read_from_mat('ground.mat')['ground']

    train_data_in = observed[:,:,0:22]
    train_data_out = ground[:,:,0:22]

    test_data_in = observed[:,:,23:]
    test_data_out = ground[:,:,23:]

    with tf.Session() as sess:
        srcnn = SRCNN(sess,train_data_in,train_data_out,test_data_in,test_data_out,learning_rate=1e-3,image_size_y=476,image_size_x=475,num_epochs=10000)
        srcnn.train()
        srcnn.test()

if __name__ == '__main__':
  tf.app.run()
