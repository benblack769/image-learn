import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
import sys
mnist = tf.keras.datasets.mnist
from sklearn import svm    			# To fit the svm classifier\

BATCH_SIZE = 64 #must be at least 2, and bigger is better.

IMAGE_WIDTH = 28
#IMAGE_SIZE = IMAGE_WIDTH*IMAGE_WIDTH

IMAGE_CHANNELS = 1
PRE_OUT_DIM = 64
PRE_CONV_SIZE = 5 # 2 in each direction
PRE_STRIDE_SIZE = 1 # consider making 2

FIRST_CONV_SIZE = 5# 2 in each direction
FIRST_STRIDE_SIZE = 2
NUM_CAPS_FIRST_LAYER = 12
CAPS_SIZE_FIRST_LAYER = 8
TOT_FIRST_SIZE = NUM_CAPS_FIRST_LAYER * CAPS_SIZE_FIRST_LAYER

NUM_CAPS_SECOND_LAYER = 13
CAPS_SIZE_SECOND_LAYER = 9
TOT_SECOND_SIZE = NUM_CAPS_SECOND_LAYER * CAPS_SIZE_SECOND_LAYER

CONV_STRATEGY = "VALID" # "VALID" or "SAME"

NUM_MATCHES = 34
NUM_MISMATCHES = 33

DISCRIM_HIDDEN_SIZE = 32
DISCRIM_OUTPUT_SIZE = 16

ADAM_learning_rate = 0.0002

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

def make_layer(name,shape):
    tot_size = reduce(mul, shape, 1)
    print(name,tot_size)
    rand_weight_vals = 0.1*np.random.randn(tot_size).astype('float32')/(shape[-1]**(0.5**0.5))
    rand_weight = np.reshape(rand_weight_vals,shape)
    return tf.Variable(rand_weight,name=name)

def squash_last_dim(arr5d):
    #sqr_size = tf.reduce_sum(arr5d * arr5d,axis=4)
    #sqr_size = tf.reshape(sqr_size,sqr_size.shape.as_list()+[1])
    #epsilon = np.float32(10e-7)
    return arr5d#(sqr_size / (np.float32(1.0) + sqr_size)) * (arr5d / tf.maximum(tf.sqrt(sqr_size),epsilon))

def make_caps_weights(shape):
    return tf.ones(shape)

class ManyMatricies:
    def __init__(self,name,num_inputs, num_outputs, input_size, output_size):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_size = input_size
        self.output_size = output_size
        self.mattensor = make_layer(name, (num_inputs, num_outputs, input_size, output_size))

    def select_matricies(self,matindicies_pairs):
        '''
        args: matindicies_pairs should have a (2, n) shape
        input matrix indicies in x[0], output in x[1]
        '''
        flat_mattenor = tf.reshape(self.mattensor,(self.num_inputs*self.num_outputs,self.input_size,self.output_size))
        flat_matindiices = matindicies_pairs[0] * self.num_outputs + matindicies_pairs[1]
        with tf.Session() as sess:
            print("hithere",self.num_outputs)
            print("argvar",sess.run(flat_matindiices))
            print("argvar",sess.run(matindicies_pairs[0]))
            print("argvar",sess.run(matindicies_pairs[1]))
        return tf.gather(flat_mattenor,flat_matindiices,axis=0)

    def multiply_selection(self,matindicies_pairs,matrix):
        # see above docs for matindicies_pairs input format
        sel_matr = self.select_matricies(matindicies_pairs)
        mul_matricies = tf.einsum("aij,ai->aj",sel_matr,matrix)
        return mul_matricies


def discriminate_fc(parent_capsule_batch,child_capsule_batch,match_forward_fns):
    '''
    makes a discrimination loss function based off different items in the batch

    ASSUMPTION: every child is connected to every parent
    '''
    batch_size = parent_capsule_batch.shape[0]
    parent_caps_locs = parent_capsule_batch.shape[1]
    parent_num_capsules = parent_capsule_batch.shape[2]
    child_num_capsules = child_capsule_batch.shape[1]
    par_caps_size = parent_capsule_batch.shape[3]
    child_caps_size = child_capsule_batch.shape[2]

    combined_parents = tf.reshape(parent_capsule_batch,(parent_num_capsules*batch_size*parent_caps_locs,par_caps_size))
    combined_children = tf.reshape(child_capsule_batch,(child_num_capsules*batch_size,child_caps_size))

    mismatch_parent_batches = tf.random_uniform((NUM_MISMATCHES,),dtype=tf.int32,minval=0,maxval=batch_size)
    mismatch_parent_locs = tf.random_uniform((NUM_MISMATCHES,),dtype=tf.int32,minval=0,maxval=parent_caps_locs)
    mismatch_child_batches = tf.random_uniform((NUM_MISMATCHES,),dtype=tf.int32,minval=0,maxval=batch_size)
    mismatch_parent_offsets = tf.random_uniform((NUM_MISMATCHES,),dtype=tf.int32,minval=0,maxval=parent_num_capsules)
    mismatch_child_offsets = tf.random_uniform((NUM_MISMATCHES,),dtype=tf.int32,minval=0,maxval=child_num_capsules)
    mismatch_parent_ids = (mismatch_parent_batches*parent_caps_locs + mismatch_parent_locs)*parent_num_capsules + mismatch_parent_offsets
    mismatch_child_ids = mismatch_child_batches*child_num_capsules + mismatch_child_offsets

    match_batches = tf.random_uniform((NUM_MATCHES,),dtype=tf.int32,minval=0,maxval=batch_size)
    match_parent_locs = tf.random_uniform((NUM_MATCHES,),dtype=tf.int32,minval=0,maxval=parent_caps_locs)
    match_parent_offsets = tf.random_uniform((NUM_MATCHES,),dtype=tf.int32,minval=0,maxval=parent_num_capsules)
    match_child_offsets = tf.random_uniform((NUM_MATCHES,),dtype=tf.int32,minval=0,maxval=child_num_capsules)
    match_parent_ids = (match_batches*parent_caps_locs + match_parent_locs)*parent_num_capsules + match_parent_offsets
    match_child_ids = match_batches*child_num_capsules + match_child_offsets
    all_parent_ids = tf.concat([mismatch_parent_ids,match_parent_ids],axis=0)
    all_child_ids = tf.concat([mismatch_child_ids,match_child_ids],axis=0)

    all_parents = tf.gather(combined_parents,all_parent_ids,axis=0)
    all_children = tf.gather(combined_children,all_child_ids,axis=0)

    match_value = tf.concat([tf.zeros(NUM_MISMATCHES),tf.ones(NUM_MATCHES)],axis=0)

    all_parent_offsets = tf.concat([mismatch_parent_offsets,match_parent_offsets],axis=0)
    all_child_offsets = tf.concat([mismatch_child_offsets,match_child_offsets],axis=0)
    mat_sel = tf.stack([all_child_offsets, all_parent_offsets])
    print(all_parent_ids.shape)
    print(mat_sel.shape)
    res_vecs = match_forward_fns.multiply_selection(mat_sel,all_children)

    logit_assignment = tf.nn.sigmoid(tf.reduce_mean(all_parents * res_vecs,axis=1))
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_assignment,labels=match_value)
    loss = tf.reduce_mean(cost)

    return loss

def learn_fn():
    lay21_match_fns = ManyMatricies("lay2-1_match_fns", NUM_CAPS_SECOND_LAYER, NUM_CAPS_FIRST_LAYER, CAPS_SIZE_SECOND_LAYER, CAPS_SIZE_FIRST_LAYER)

    in_img = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))

    '''ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.repeat(count=10000000000000)
    ds = ds.shuffle(len(x_train))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.map(lambda batch: tf.reshape(batch,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS)))
    ds = ds.prefetch(8)
    iter = ds.make_one_shot_iterator()
    in_img = iter.get_next()'''


    pre_conv = make_layer("pre_conv",(PRE_CONV_SIZE, PRE_CONV_SIZE, IMAGE_CHANNELS, PRE_OUT_DIM))
    pre_out = tf.nn.relu(tf.nn.conv2d(in_img,pre_conv,(1,PRE_STRIDE_SIZE,PRE_STRIDE_SIZE,1),CONV_STRATEGY))

    lay1conv = make_layer("lay1conv",(FIRST_CONV_SIZE, FIRST_CONV_SIZE, PRE_OUT_DIM, TOT_FIRST_SIZE))
    print(lay1conv.shape)
    lay1conv_out = tf.nn.conv2d(pre_out,lay1conv,(1,FIRST_STRIDE_SIZE,FIRST_STRIDE_SIZE,1),CONV_STRATEGY)
    conv_size = lay1conv_out.shape[1]
    capsuled_lay1 = tf.reshape(lay1conv_out,(BATCH_SIZE, conv_size, conv_size, NUM_CAPS_FIRST_LAYER, CAPS_SIZE_FIRST_LAYER))
    squashed_lay1 = squash_last_dim(capsuled_lay1)
    capsule_weights = make_caps_weights((BATCH_SIZE, conv_size, conv_size, NUM_CAPS_FIRST_LAYER, 1))
    lay1caps_mask = tf.ones((BATCH_SIZE, conv_size, conv_size, NUM_CAPS_FIRST_LAYER, 1))
    lay1caps_mask = tf.nn.dropout(lay1caps_mask,0.2,noise_shape=(1,1,1,NUM_CAPS_FIRST_LAYER,1))
    #with tf.Session() as sess:
    #    print("mask ",sess.run(mask))
    #lay1caps_mask = tf.reshape(lay1caps_mask,(BATCH_SIZE,conv_size*conv_size,NUM_CAPS_FIRST_LAYER,1))
    weighted_lay1 = squashed_lay1 * capsule_weights# * lay1caps_mask

    lay2fullconect = make_layer("lay2fullconect",(TOT_FIRST_SIZE*conv_size*conv_size,TOT_SECOND_SIZE))
    combined_lay1 = tf.reshape(weighted_lay1,(BATCH_SIZE, TOT_FIRST_SIZE*conv_size*conv_size))
    lay2_raw = tf.matmul(combined_lay1,lay2fullconect)
    lay2_capsules = tf.reshape(lay2_raw,(BATCH_SIZE,NUM_CAPS_SECOND_LAYER,CAPS_SIZE_SECOND_LAYER))

    capsuled_lay1 = tf.reshape(capsuled_lay1,(BATCH_SIZE,conv_size*conv_size,NUM_CAPS_FIRST_LAYER,CAPS_SIZE_FIRST_LAYER))
    #print(capsuled_lay1.shape)
    #print(lay2_capsules.shape)
    #exit(1)
    #capsuled_lay1 = capsuled_lay1 * mask
    loss = discriminate_fc(capsuled_lay1,lay2_capsules,lay21_match_fns)
    optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    optim = optimizer.minimize(loss)

    train_data = np.copy(x_train)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #loss_val,opt_val = sess.run([loss,optim])
        while True:
            for x in range(3):
                np.random.shuffle(train_data)
                loss_sum = 0
                train_count = 0
                for x in range(5000):
                    loss_val,opt_val = sess.run([loss,optim],feed_dict={
                        in_img: np.reshape(train_data[x:x+BATCH_SIZE],(BATCH_SIZE,IMAGE_WIDTH, IMAGE_WIDTH, 1))
                    })
                    loss_sum += loss_val
                    train_count += 1
                #print(np.any(np.isnan(caps1)))
                print("loss: {}".format(loss_sum/train_count))
            all_test_out = []
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
            #print(class_out)
            sys.stdout.flush()

learn_fn()
