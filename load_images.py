import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
import sys
mnist = tf.keras.datasets.mnist
from sklearn import svm    			# To fit the svm classifier\

BATCH_SIZE = 8 #must be at least 2, and bigger is better.

IMAGE_WIDTH = 28
#IMAGE_SIZE = IMAGE_WIDTH*IMAGE_WIDTH

IMAGE_CHANNELS = 1
CONV_STRATEGY = "SAME" # "VALID" or "SAME"

ADAM_learning_rate = 0.001

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

def make_layer(name,shape):
    tot_size = reduce(mul, shape, 1)
    print(name,tot_size)
    rand_weight_vals = np.random.randn(tot_size).astype('float32')/(shape[-1]**(0.5**0.5))
    rand_weight = np.reshape(rand_weight_vals,shape)
    return tf.Variable(rand_weight,name=name)

def learn_fn():
    in_img = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))
    mask = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, 1))
    feed_input = tf.concat([in_img*mask,mask],axis=3)

    CONV1_SIZE = 5
    LAY1_SIZE = 32
    conv1_weights = make_layer("conv1_weights",(CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS+1, LAY1_SIZE))
    lay1_outs = tf.nn.relu(tf.nn.conv2d(feed_input,conv1_weights,(1,1,1,1),CONV_STRATEGY))

    CONV2_SIZE = 3
    lay2_weights = make_layer("lay2_weights",(CONV2_SIZE, CONV2_SIZE, LAY1_SIZE, LAY1_SIZE))
    lay2_outs = tf.nn.relu(tf.nn.conv2d(feed_input,lay2_weights,(1,1,1,1),CONV_STRATEGY))

    lay3_weights = make_layer("lay3_weights",(1, 1, LAY1_SIZE, LAY1_SIZE))
    lay3_outs = tf.nn.relu(tf.nn.conv2d(lay2_outs,lay3_weights,(1,1,1,1),CONV_STRATEGY))

    fin_weights = make_layer("fin_weights",(1, 1, LAY1_SIZE, 1))

    fin_outs = tf.sigmoid(0.01*tf.nn.conv2d(lay3_outs,fin_weights,(1,1,1,1),CONV_STRATEGY))
    fin_outs = tf.nn.sigmoid_cross_entropy_with_logits()
    loss = tf.reduce_mean(fin_outs)

    first_optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    first_optim = first_optimizer.minimize(loss)

    next_input = tf.stop_gradient(first_layer_out)


    train_data = np.copy(x_train)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #loss_val,opt_val = sess.run([loss,optim])
        while True:
            for x in range(20):
                np.random.shuffle(train_data)
                loss_sum = 0
                train_count = 0
                for x in range(100):
                    loss_val,opt_val = sess.run([first_loss,first_optim],feed_dict={
                        in_img: np.reshape(train_data[x:x+BATCH_SIZE],(BATCH_SIZE,IMAGE_WIDTH, IMAGE_WIDTH, 1))
                    })
                    loss_sum += loss_val
                    train_count += 1
                #print(np.any(np.isnan(caps1)))
                print("loss: {}".format(loss_sum/train_count))
            sys.stdout.flush()

learn_fn()
