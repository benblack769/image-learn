import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
import sys
import os
import shutil
import random
from PIL import Image
mnist = tf.keras.datasets.mnist
from sklearn import svm    			# To fit the svm classifier
from loss_ballancing import TensorLossBallancer
from sample_queue import SampleStorage

BATCH_SIZE = 32

IMAGE_WIDTH = 28
#IMAGE_SIZE = IMAGE_WIDTH*IMAGE_WIDTH

PROB_OF_MASK_ONE = 0.01

SAMPLE_MAX = 5000

DROPOUT_CHANNELS = 1

CALC_ITERS = 8

IMAGE_CHANNELS = 1
CONV_STRATEGY = "SAME" # "VALID" or "SAME"

SELECTION_SIZE = 16 # number of samples for an expectation training
INFO_BUFFER_SIZE = 64

UNIFORM_SAMPLE_PROB = 0.005 # baseline probabiltiy it will sample to allow unexpected results to occur

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

def unpool(arr4d,pool_size):
    shape = arr4d.get_shape().as_list()
    newshape = [shape[0],shape[1]*pool_size,shape[2]*pool_size,shape[3]]
    reshaped = tf.reshape(arr4d,[shape[0],shape[1],1,shape[2],1,shape[3]])
    tiled = tf.tile(reshaped,(1,1,pool_size,1,pool_size,1))
    return tf.reshape(tiled,newshape)


def test_unpool():
    arr = np.arange(3*4*4*2).reshape((3,4,4,2))
    test_val = tf.constant(arr,shape=(3,4,4,2))
    res = unpool(test_val,2)
    with tf.Session() as sess:
        print(sess.run(res)[0])


def unpool2x2(input,orig_shape):
    base_shape = input.get_shape().as_list()
    batch_size = tf.shape(input)[0]
    reshaped_pool = tf.reshape(input,[batch_size,base_shape[1],base_shape[2],1,base_shape[3]])
    x_concatted = tf.concat([reshaped_pool,reshaped_pool],3)
    x_reshaped = tf.reshape(x_concatted,[batch_size,base_shape[1],1,base_shape[2]*2,base_shape[3]])
    y_concatted = tf.concat([x_reshaped,x_reshaped],2)
    y_reshaped = tf.reshape(y_concatted,[batch_size,base_shape[1]*2,base_shape[2]*2,base_shape[3]])
    #sliced = y_reshaped[:,:orig_shape[1],:orig_shape[2]]
    sliced = tf.slice(y_reshaped,[0,0,0,0],[batch_size,orig_shape[1],orig_shape[2],base_shape[3]])
    return sliced
#test_unpool()
#exit(1)

def reconstruction_loss(model_reconstruction, input_img):
    pix_wise_losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_reconstruction,labels=input_img)
    flat_loss = tf.reshape(pix_wise_losses,(BATCH_SIZE, IMAGE_WIDTH * IMAGE_WIDTH))
    return tf.reduce_mean(flat_loss,axis=1)

def mask_loss(actual_win,actual_mask_sel,mask_sel_probs):
    calced_win_prob = tf.reduce_sum(actual_mask_sel * mask_sel_probs,axis=[1,2])
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=calced_win_prob,labels=actual_win)


def lay_pool_skip_method(input):
    lay1size = 32
    CONV1_SIZE=[3,3]
    POOL_SIZE=[2,2]
    POOL_STRIDES=[2,2]
    DEPTH=5
    basic_outs = []
    orig_reduction = tf.layers.dense(
        inputs=input,
        units=lay1size,
        activation=tf.nn.relu
    )
    cur_out = orig_reduction
    for x in range(DEPTH):
        lay1_outs = tf.layers.conv2d(
            inputs=cur_out,
            filters=lay1size,
            kernel_size=CONV1_SIZE,
            padding="same",
            activation=tf.nn.relu)
        lay_1_pool = tf.layers.average_pooling2d(
            inputs=lay1_outs,
            pool_size=POOL_SIZE,
            strides=POOL_STRIDES,
            padding='same',
        )
        basic_outs.append(lay1_outs)
        cur_out = lay_1_pool
        print(lay_1_pool.shape)

    old_val = basic_outs[DEPTH-1]
    for y in range(DEPTH-2,-1,-1):
        skip_val = basic_outs[y]
        depooled = unpool2x2(old_val,skip_val.get_shape().as_list())
        base_val = depooled + skip_val

        base_val = tf.layers.conv2d(
            inputs=base_val,
            filters=lay1size,
            kernel_size=CONV1_SIZE,
            padding="same",
            activation=tf.nn.relu)
        old_val = base_val
        print(depooled.shape,)


    combined_input = old_val+orig_reduction
    refine_layer1 = tf.layers.dense(
        inputs=combined_input,
        units=lay1size,
        activation=tf.nn.relu
    )
    generate_out = tf.layers.dense(
        inputs=refine_layer1,
        units=1
    )
    '''refine_mask1 = tf.layers.dense(
        inputs=combined_input,
        units=lay1size,
        activation=tf.nn.relu
    )
    mask_out = tf.layers.dense(
        inputs=refine_layer1,
        units=1
    )'''
    return (generate_out)

def train_generator():
    train_data = np.copy(x_train)
    for iteration_run in range(1000000000):
        np.random.shuffle(train_data)
        for x in range(0,len(train_data),BATCH_SIZE):
            yield np.reshape(train_data[x:x+BATCH_SIZE],(BATCH_SIZE,IMAGE_WIDTH,IMAGE_WIDTH,1))

def sample_prob(mask_p):
    # pick better masks with higher probabilty by squaring
    prob = mask_p# * mask_p
    #print(prob)
    prob = prob / (np.sum(prob))
    return prob

def mask_sample(mask_probs_batch):
    new_masks = []
    for b in range(BATCH_SIZE):
        mask_p = mask_probs_batch[b]
        flat_mask_p = mask_p.reshape((IMAGE_WIDTH*IMAGE_WIDTH,))
        normalized_flat_mask = sample_prob(flat_mask_p)
        mask_val = np.random.choice(np.arange(IMAGE_WIDTH*IMAGE_WIDTH),p=normalized_flat_mask)
        new_flat_mask = np.zeros((IMAGE_WIDTH*IMAGE_WIDTH,))
        new_flat_mask[mask_val] = 1.0
        new_mask = np.reshape(new_flat_mask,(IMAGE_WIDTH,IMAGE_WIDTH,1))
        new_masks.append(new_mask)
    return np.stack(new_masks)

def get_input(img,mask):
    return tf.concat([img*mask,mask],axis=-1)

class AI_Agent:
    def __init__(self):
        self.in_img = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))
        self.dropout_mask = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS))

        self.is_actual_winner = tf.placeholder(tf.float32, (BATCH_SIZE//2, )) # one for max selection 0 for everything else
        self.mask_actual_selection = tf.placeholder(tf.float32, (BATCH_SIZE//2, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS)) # one for selected vars, 0 for everything else

        self.is_actual_winner = tf.concat([self.is_actual_winner,tf.zeros_like(self.is_actual_winner)],axis=0)
        self.mask_actual_selection = tf.concat([self.mask_actual_selection,tf.zeros_like(self.mask_actual_selection)],axis=0)
        self.is_actual_winner = tf.reshape(self.is_actual_winner, (BATCH_SIZE, 1))

        mask_loss_mask = tf.concat([tf.ones(BATCH_SIZE//2),tf.zeros(BATCH_SIZE//2)],axis=0)
        gen_loss_mask = tf.concat([tf.zeros(BATCH_SIZE//2),tf.ones(BATCH_SIZE//2)],axis=0)

        #feed_input = tf.concat([in_img*mask,(mask)],axis=3)
        #AdamOptimizer
        #RMSPropOptimizer
        info_optimizer = tf.train.RMSPropOptimizer(learning_rate=ADAM_learning_rate)
        #mask_optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)

        model_input = get_input(self.in_img, self.dropout_mask)
        mask_expectation_logits = lay_pool_skip_method(model_input)
        self.reconstruction = lay_pool_skip_method(model_input)
        self.reconstruct_loss_batchwise = reconstruction_loss(self.reconstruction,self.in_img)
        self.tot_reconstr_loss = tf.reduce_mean(self.reconstruct_loss_batchwise*gen_loss_mask)
        self.sigmoid_reconstr = tf.sigmoid(self.reconstruction)

        mask_losses = mask_loss(self.is_actual_winner,self.mask_actual_selection,mask_expectation_logits)
        self.mask_loss = tf.reduce_mean(mask_losses*mask_loss_mask)
        self.generated_mask_expectations = tf.nn.sigmoid(mask_expectation_logits)

        self.loss_ballancer = TensorLossBallancer(0.95,0.99)
        adj_mask_loss, adj_reconstr_loss, self.stateful_adj = self.loss_ballancer.adjust(self.mask_loss,self.tot_reconstr_loss)
        combined_ballanced_loss = adj_mask_loss+adj_reconstr_loss
        self.optim = info_optimizer.minimize(combined_ballanced_loss)
        self.gen_train_sampler = SampleStorage(SAMPLE_MAX)
        self.mask_train_sampler = SampleStorage(SAMPLE_MAX)

    def calc_games_batch(self,sess,input_img_batch):
        dropout_mask = np.zeros((BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS))

        all_masks = []
        all_mask_actuals = []
        all_mask_additions = []
        for i in range(CALC_ITERS):
            mask_probs = sess.run(self.generated_mask_expectations,feed_dict={
                self.in_img: input_img_batch,
                self.dropout_mask: dropout_mask,
            })
            single_mask_sample = mask_sample(mask_probs)
            all_masks.append(dropout_mask)
            dropout_mask = np.logical_or(dropout_mask,single_mask_sample)
            all_mask_additions.append(single_mask_sample)

        reconstr_sig,reconstruct_loss = sess.run([self.sigmoid_reconstr,self.reconstruct_loss_batchwise],feed_dict={
            self.in_img:input_img_batch,
            self.dropout_mask:dropout_mask,
        })
        return reconstruct_loss,reconstr_sig,dropout_mask,all_masks,all_mask_additions

    def add_gen_train(self,input,mask):
        self.gen_train_sampler.put_list(zip(input,mask))

    def add_mask_train(self,input,prev_mask,mask_sel,does_win):
        self.mask_train_sampler.put_list(zip(input,prev_mask,mask_sel,does_win))

    def sample_mask(self):
        data = [self.mask_train_sampler.sample() for _ in range(BATCH_SIZE//2)]
        inputs = [d[0] for d in data]
        input_mask = [d[1] for d in data]
        mask_selection = [d[2] for d in data]
        does_win = [d[3] for d in data]
        return np.stack(inputs),np.stack(input_mask),np.stack(mask_selection),np.stack(does_win)

    def sample_gen(self):
        data = [self.gen_train_sampler.sample() for _ in range(BATCH_SIZE//2)]
        inputs = [d[0] for d in data]
        mask = [d[1] for d in data]
        return np.stack(inputs),np.stack(mask)

    def train_from_stored(self,sess):
        m_inp,m_mask,m_mask_sel,m_does_win = self.sample_mask()
        g_inp,g_mask = self.sample_gen()

        input = np.concatenate([m_inp,g_inp],axis=0)
        mask = np.concatenate([m_mask,g_mask],axis=0)

        # we don't have compares for the other part of the batch, so just add zerso to fill in the tensors
        act_does_win = np.concatenate([m_does_win,np.zeros_like(m_does_win)],axis=0)
        act_does_win = np.reshape(act_does_win,(BATCH_SIZE,1))
        mask_sel = np.concatenate([m_mask_sel,np.zeros_like(m_mask_sel)],axis=0)

        mask_loss,reconst_loss,opt,stateful_adj = sess.run([self.mask_loss,self.tot_reconstr_loss,self.optim,self.stateful_adj],feed_dict={
            self.in_img: input,
            self.dropout_mask: mask,
            self.is_actual_winner: act_does_win,
            self.mask_actual_selection: mask_sel,
        })
        #print(sess.run(self.loss_ballancer.a_adj_state),sess.run(self.loss_ballancer.b_adj_state))
        #print()
        return mask_loss,reconst_loss
        #print("{}\t{}".format(mask_loss,reconst_loss))


def eval_winners(losses1,losses2):
    win_val = np.asarray([(1 if l1 > l2 else 0) for l1,l2 in zip(losses1,losses2)],dtype=np.float32)
    return win_val,1.0-win_val

def add_train_all(ai,input,fin_mask,does_win,in_masks,mask_adds):
    ai.add_gen_train(input,fin_mask)
    for in_mask,mask_add in zip(in_masks,mask_adds):
        ai.add_mask_train(input,in_mask,mask_add,does_win)

def calc_pair_add_train(sess,ai1,ai2,input):
    reconstr_loss1,_,final_mask1,all_masks1,all_masks_adds1 = ai1.calc_games_batch(sess,input)
    reconstr_loss2,_,final_mask2,all_masks2,all_masks_adds2 = ai2.calc_games_batch(sess,input)

    winner1,winner2 = eval_winners(reconstr_loss1,reconstr_loss2)

    add_train_all(ai1,input,final_mask1,winner1,all_masks1,all_masks_adds1)
    add_train_all(ai2,input,final_mask2,winner2,all_masks2,all_masks_adds2)

def save_image(filename,float_data):
    img_data = (float_data * 255.0).astype(np.uint8)
    img = Image.fromarray(img_data,mode="L")
    img.save(filename)

class CSV_gen:
    def __init__(self, header):
        self.header = header
        self.sorted_header = list(sorted(self.header))
        self.rows = {h:[] for h in header}

    def add_row(self,row_data):
        assert list(sorted(row_data.keys())) == self.sorted_header
        for h,d in row_data.items():
            self.rows[h].append(d)

def calc_pair_save_imgs(sess,ais,input,folder):
    NUM_INPUT_SAVES = 3

    for x in range(NUM_INPUT_SAVES):
        foldname = folder+str(x)+"/"
        os.mkdir(foldname)
        save_image(foldname+"input.jpg",np.squeeze(input[x]))

    for ai_idx,ai in enumerate(ais):
        reconstr_loss1,reconstr1,final_mask1,all_masks1,all_masks_adds1 = ai.calc_games_batch(sess,input)
        for x in range(NUM_INPUT_SAVES):
            foldname = folder + str(x)+"/"
            save_image(foldname+"recon{}.jpg".format(ai_idx),np.squeeze(reconstr1[x]))
            save_image(foldname+"mask{}.jpg".format(ai_idx),np.squeeze(final_mask1[x]))

def train_all_ais():
    batch_generator = train_generator()

    NUM_AIS = 3
    NUM_ITERS = 10
    all_ais = [AI_Agent() for _ in range(NUM_AIS)]
    BASEFOLDER = "generated/"

    if os.path.exists(BASEFOLDER):
        shutil.rmtree(BASEFOLDER)
    os.mkdir(BASEFOLDER)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # initial population
        for x in range(10):
            ai1 = random.choice(all_ais)
            ai2 = random.choice(all_ais)

            calc_pair_add_train(sess,ai1,ai2,next(batch_generator))

        for x in range(1000000000):
            for y in range(NUM_AIS):
                ai1 = random.choice(all_ais)
                ai2 = random.choice(all_ais)

                calc_pair_add_train(sess,ai1,ai2,next(batch_generator))

            for ai in all_ais:
                tot_mask_loss, tot_reconst_los = 0,0
                for y in range(NUM_ITERS):
                    mask_loss,reconst_loss = ai.train_from_stored(sess)
                    tot_mask_loss += mask_loss
                    tot_reconst_los += reconst_loss

                print("{}\t{}".format(tot_mask_loss/NUM_ITERS,tot_reconst_los/NUM_ITERS))

            if x % 20 == 0:
                foldname = BASEFOLDER+str(x)+"/"
                os.mkdir(foldname)
                calc_pair_save_imgs(sess,all_ais,next(batch_generator),foldname)


def init_mask():
    return np.zeros((BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS),dtype=np.float32)

def norm_probs(x):
    return x * (np.float32(1.0) / np.sum(x))

def sample_probs(size,probs):
    cumu = np.add.accumulate(probs)
    val = np.random.uniform(size=size)
    res = np.digitize(val,cumu)
    return res
    #return np.random.choice(size,p=probs)


train_all_ais()
