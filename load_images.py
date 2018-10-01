import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
import sys
import os
from PIL import Image
mnist = tf.keras.datasets.mnist
from sklearn import svm    			# To fit the svm classifier\

BATCH_SIZE = 8 #must be at least 2, and bigger is better.

IMAGE_WIDTH = 28
#IMAGE_SIZE = IMAGE_WIDTH*IMAGE_WIDTH

PROB_OF_MASK_ONE = 0.01

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

def rand_prob(size,prob):
    mymask = np.random.uniform(size=size).astype(np.float32)
    mymask = 1.0-np.minimum(1.0,np.floor(mymask*(1.0/prob)))
    return mymask

def sqr(x):
    return x * x

DROPOUT_CHANNELS = 9

def learn_fn():
    in_img = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))
    dropout_mask = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS))
    #feed_input = tf.concat([in_img*mask,(mask)],axis=3)

    CONV1_SIZE = 5
    LAY1_SIZE = 64
    conv1_weights = make_layer("conv1_weights",(CONV1_SIZE, CONV1_SIZE, IMAGE_CHANNELS, LAY1_SIZE))
    lay1_outs = tf.nn.relu(tf.nn.conv2d(in_img,conv1_weights,(1,1,1,1),CONV_STRATEGY))

    CONV2_SIZE = 3
    lay2_weights = make_layer("lay2_weights",(CONV2_SIZE, CONV2_SIZE, LAY1_SIZE, LAY1_SIZE))
    lay2_outs = tf.nn.relu(tf.nn.conv2d(lay1_outs,lay2_weights,(1,1,1,1),CONV_STRATEGY))

    DROPLAY_CONV_SIZE = 3
    DROPLAY_CHANNEL_SIZE = 8
    DROPLAY_OUT_SIZE = (DROPLAY_CHANNEL_SIZE+1)*DROPOUT_CHANNELS
    droplay_weights = make_layer("droplay_weights",(DROPLAY_CONV_SIZE, DROPLAY_CONV_SIZE, LAY1_SIZE, DROPLAY_CHANNEL_SIZE*DROPOUT_CHANNELS))
    droplay_outs = tf.nn.conv2d(lay2_outs,droplay_weights,(1,1,1,1),CONV_STRATEGY)
    droplay_outs = tf.reshape(droplay_outs,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS, DROPLAY_CHANNEL_SIZE))
    broadcastable_dmask = tf.reshape(dropout_mask,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS, 1))
    droplay_outs = droplay_outs * broadcastable_dmask
    droplay_outs = tf.reshape(droplay_outs,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS* DROPLAY_CHANNEL_SIZE))
    tot_droplay_outs = tf.concat([droplay_outs,dropout_mask],axis=3)

    CONV3_SIZE = 5
    lay3_weights = make_layer("lay3_weights",(CONV3_SIZE, CONV3_SIZE, DROPLAY_OUT_SIZE, LAY1_SIZE))
    lay3_outs = tf.nn.relu(tf.nn.conv2d(tot_droplay_outs,lay3_weights,(1,1,1,1),CONV_STRATEGY))

    CONV4_SIZE = 5
    lay4_weights = make_layer("lay4_weights",(CONV4_SIZE, CONV4_SIZE, LAY1_SIZE, LAY1_SIZE))
    lay4_outs = tf.nn.relu(tf.nn.conv2d(lay3_outs,lay4_weights,(1,1,1,1),CONV_STRATEGY))

    fin_weights = make_layer("fin_weights",(1, 1, LAY1_SIZE, 1))

    fin_outs = tf.sigmoid(0.01*tf.nn.conv2d(lay4_outs,fin_weights,(1,1,1,1),CONV_STRATEGY))

    fin_out_lossess = sqr(in_img-fin_outs)
    #fin_out_lossess = tf.nn.sigmoid_cross_entropy_with_logits(labels=in_img,logits=fin_outs)
    loss = tf.reduce_mean(fin_out_lossess)

    optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    optim = optimizer.minimize(loss)

    #next_input = tf.stop_gradient(first_layer_out)

    train_data = np.copy(x_train)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #loss_val,opt_val = sess.run([loss,optim])
        for iteration_run in range(100000):
            for x in range(10):
                np.random.shuffle(train_data)
                loss_sum = 0
                train_count = 0
                for x in range(100):
                    #exit(1)
                    loss_val,opt_val = sess.run([loss,optim],feed_dict={
                        in_img: np.reshape(train_data[x*BATCH_SIZE:(x+1)*BATCH_SIZE],(BATCH_SIZE,IMAGE_WIDTH, IMAGE_WIDTH, 1)),
                        dropout_mask: rand_prob((BATCH_SIZE,IMAGE_WIDTH,IMAGE_WIDTH,DROPOUT_CHANNELS),PROB_OF_MASK_ONE),
                    })
                    loss_sum += loss_val
                    train_count += 1
                #print(np.any(np.isnan(caps1)))
                print("loss: {}".format(loss_sum/train_count))
            sys.stdout.flush()

            save_folder = "examples/run_{}/".format(iteration_run)
            os.makedirs(save_folder,exist_ok=True)
            for x in range(3):
                batch_data = x_test[x*BATCH_SIZE:(x+1)*BATCH_SIZE]
                img_data = sess.run(fin_outs,feed_dict={
                    in_img: np.reshape(batch_data,(BATCH_SIZE,IMAGE_WIDTH, IMAGE_WIDTH, 1)),
                    dropout_mask: rand_prob((BATCH_SIZE,IMAGE_WIDTH,IMAGE_WIDTH,DROPOUT_CHANNELS),PROB_OF_MASK_ONE),
                })
                img_data = np.reshape(img_data,(BATCH_SIZE,IMAGE_WIDTH,IMAGE_WIDTH))
                for y in range(BATCH_SIZE):
                    img_name = str(x*BATCH_SIZE+y)+".png"
                    image_vec = (img_data[y]*255.0).astype(np.uint8)
                    img = Image.fromarray(image_vec,mode="L")
                    img.save(save_folder+img_name)

learn_fn()
