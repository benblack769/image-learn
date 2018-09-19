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
PRE_OUT_DIM = 64
PRE_CONV_SIZE = 5 # 2 in each direction
PRE_STRIDE_SIZE = 1 # consider making 2

FIRST_CONV_SIZE = 5# 2 in each direction
FIRST_STRIDE_SIZE = 1
NUM_CAPS_FIRST_LAYER = 4
CAPS_SIZE_FIRST_LAYER = 16
TOT_FIRST_SIZE = NUM_CAPS_FIRST_LAYER * CAPS_SIZE_FIRST_LAYER

NUM_CAPS_SECOND_LAYER = 5
CAPS_SIZE_SECOND_LAYER = 24
TOT_SECOND_SIZE = NUM_CAPS_SECOND_LAYER * CAPS_SIZE_SECOND_LAYER

CONV_STRATEGY = "VALID" # "VALID" or "SAME"

NUM_MATCHES = 128
NUM_MISMATCHES = 128

DISCRIM_HIDDEN_SIZE = 32
DISCRIM_OUTPUT_SIZE = 16

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

def make_gather_conv(conv_size,input_size):
    res = []
    conv_size_sqr = conv_size*conv_size
    for x in range(conv_size):
        for y in range(conv_size):
            for i in range(input_size):
                for j in range(input_size*conv_size_sqr):
                    res.append(int(j // input_size == x * conv_size + y))
    res_np = np.asarray(res,dtype=np.float32)
    res_reshaped = np.reshape(res_np,(conv_size,conv_size,input_size,conv_size*conv_size*input_size))
    return tf.constant(res_reshaped,dtype=tf.float32)

def learn_fn():
    in_img = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))

    '''ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.repeat(count=10000000000000)
    ds = ds.shuffle(len(x_train))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda batch: tf.reshape(batch,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS)))
    ds = ds.prefetch(8)
    iter = ds.make_one_shot_iterator()
    in_img = iter.get_next()'''


    pre_conv_through = make_layer("pre_conv_through",(PRE_CONV_SIZE, PRE_CONV_SIZE, IMAGE_CHANNELS, PRE_OUT_DIM))
    pre_conv_cmp = make_layer("pre_conv_cmp",(PRE_CONV_SIZE, PRE_CONV_SIZE, IMAGE_CHANNELS, PRE_OUT_DIM))

    pre_conv_through_out = tf.nn.relu(tf.nn.conv2d(in_img,pre_conv_through,(1,1,1,1),CONV_STRATEGY))
    pre_conv_cmp_out = tf.nn.relu(tf.nn.conv2d(in_img,pre_conv_cmp,(1,1,1,1),CONV_STRATEGY))

    cmp_final = make_layer("cmp_final",(1, 1, PRE_OUT_DIM, DISCRIM_OUTPUT_SIZE))
    cmp_final_out = tf.nn.conv2d(pre_conv_cmp_out,cmp_final,(1,1,1,1),CONV_STRATEGY)

    first_layer_conv = make_layer("first_layer_conv",(FIRST_CONV_SIZE, FIRST_CONV_SIZE, PRE_OUT_DIM, TOT_FIRST_SIZE))
    first_layer_out = tf.nn.conv2d(pre_conv_through_out,first_layer_conv,(1,FIRST_STRIDE_SIZE,FIRST_STRIDE_SIZE,1),CONV_STRATEGY)
    first_layer_activ = tf.nn.relu(first_layer_out)

    first_layer_fin = make_layer("first_layer_conv", (1, 1, TOT_FIRST_SIZE, DISCRIM_OUTPUT_SIZE*FIRST_CONV_SIZE*FIRST_CONV_SIZE))
    first_layer_fin_out = tf.nn.conv2d(first_layer_activ,first_layer_fin,(1,1,1,1),CONV_STRATEGY)

    cmp_final_copy_fn = make_gather_conv(FIRST_CONV_SIZE,DISCRIM_OUTPUT_SIZE)
    copied_cmp_fin = tf.nn.conv2d(cmp_final_out,cmp_final_copy_fn,(1,1,1,1),CONV_STRATEGY)

    conv_size = first_layer_activ.shape[1]

    cmp_reshaped = tf.reshape(copied_cmp_fin,(BATCH_SIZE*conv_size*conv_size*FIRST_CONV_SIZE*FIRST_CONV_SIZE,DISCRIM_OUTPUT_SIZE))
    first_final_reshaped = tf.reshape(first_layer_fin_out,(BATCH_SIZE*conv_size*conv_size*FIRST_CONV_SIZE*FIRST_CONV_SIZE,DISCRIM_OUTPUT_SIZE))

    match_len = cmp_reshaped.shape.as_list()[0]
    avoid_match_len = match_len // BATCH_SIZE
    mismatch_batch_offset = tf.random_uniform((1,),minval=avoid_match_len,maxval=match_len-avoid_match_len,dtype=tf.int32)
    mismatch_batch_offset = tf.reshape(mismatch_batch_offset,[])
    mismatch_vals = tf.concat([first_final_reshaped[mismatch_batch_offset:],first_final_reshaped[:mismatch_batch_offset]],axis=0)

    first_vals = tf.concat([first_final_reshaped,mismatch_vals],axis=0)
    cmp_vals = tf.concat([cmp_reshaped,cmp_reshaped],axis=0)

    match_value = tf.concat([tf.ones(match_len),tf.zeros(match_len)],axis=0)

    logit_assignment = tf.nn.sigmoid(tf.reduce_mean(first_vals * cmp_vals,axis=1)*0.1)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_assignment,labels=match_value)
    loss = tf.reduce_mean(cost)

    optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    optim = optimizer.minimize(loss)

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
                    loss_val,opt_val = sess.run([loss,optim],feed_dict={
                        in_img: np.reshape(train_data[x:x+BATCH_SIZE],(BATCH_SIZE,IMAGE_WIDTH, IMAGE_WIDTH, 1))
                    })
                    loss_sum += loss_val
                    train_count += 1
                #print(np.any(np.isnan(caps1)))
                print("loss: {}".format(loss_sum/train_count))
            '''all_test_out = []
            all_train_out = []
            for x in range(0,1000,BATCH_SIZE):
                class_out = sess.run(lay2_raw,feed_dict={
                    in_img: np.reshape(x_test[x:x+BATCH_SIZE],(BATCH_SIZE,IMAGE_WIDTH, IMAGE_WIDTH, 1))
                })
                train_out = sess.run(lay2_raw,feed_dict={
                    in_img: np.reshape(x_train[x:x+BATCH_SIZE],(BATCH_SIZE,IMAGE_WIDTH, IMAGE_WIDTH, 1))
                })
                all_test_out.append(class_out)
                all_train_out.append(train_out)

            tot_test_out = np.concatenate(all_test_out,axis=0)
            tot_train_out = np.concatenate(all_train_out,axis=0)

            tot_test_cmp = y_test[:len(tot_test_out)]
            print(tot_test_cmp.shape)
            tot_train_cmp = y_train[:len(tot_train_out)]

            logit_model = svm.SVC(kernel='linear')
            logit_model.fit(tot_train_out,tot_train_cmp)
            print(tot_test_out.shape)
            print(tot_test_cmp.shape)
            score = logit_model.score(tot_test_out,tot_test_cmp)
            print("supervised score: {}".format(score))
            #print(class_out)'''
            sys.stdout.flush()

learn_fn()
