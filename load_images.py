import tensorflow as tf
import numpy as np
from operator import mul
from functools import reduce
import sys
import os
import shutil
from PIL import Image
mnist = tf.keras.datasets.mnist
from sklearn import svm    			# To fit the svm classifier\

BATCH_SIZE = 32

IMAGE_WIDTH = 28
#IMAGE_SIZE = IMAGE_WIDTH*IMAGE_WIDTH

PROB_OF_MASK_ONE = 0.01

DROPOUT_CHANNELS = 9

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

#test_unpool()
#exit(1)

def mask_info(in_img, dropout_mask):
    CONV1_SIZE = 5
    LAY1_SIZE = 64

    lay1_outs = tf.layers.conv2d(
        inputs=in_img,
        filters=LAY1_SIZE,
        kernel_size=[CONV1_SIZE, CONV1_SIZE],
        padding="same",
        activation=tf.nn.relu)

    CONV2_SIZE = 3
    lay2_outs = tf.layers.conv2d(
        inputs=lay1_outs,
        filters=LAY1_SIZE,
        kernel_size=[CONV2_SIZE, CONV2_SIZE],
        padding="same",
        activation=tf.nn.relu)

    DROPLAY_CONV_SIZE = 3
    DROPLAY_CHANNEL_SIZE = 8
    droplay_outs = tf.layers.conv2d(
        inputs=lay2_outs,
        filters=DROPLAY_CHANNEL_SIZE*DROPOUT_CHANNELS,
        kernel_size=[DROPLAY_CONV_SIZE, DROPLAY_CONV_SIZE],
        padding="same",
        activation=None)
    droplay_outs = tf.reshape(droplay_outs,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS, DROPLAY_CHANNEL_SIZE))
    broadcastable_dmask = tf.reshape(dropout_mask,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS, 1))
    droplay_outs = droplay_outs * broadcastable_dmask
    droplay_outs = tf.reshape(droplay_outs,(BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS* DROPLAY_CHANNEL_SIZE))
    tot_droplay_outs = tf.concat([droplay_outs,dropout_mask],axis=3)

    POOLED_LAYER_SIZE = 64
    DROPLAY_POOL_SIZE = 2
    pooled_outs = tf.layers.average_pooling2d(
        inputs=tot_droplay_outs,
        pool_size=[DROPLAY_POOL_SIZE,DROPLAY_POOL_SIZE],
        strides=DROPLAY_POOL_SIZE,
    )

    GEN_CONV1_SIZE = 3
    gen_lay1_outs = tf.layers.conv2d(
        inputs=pooled_outs,
        filters=POOLED_LAYER_SIZE,
        kernel_size=[GEN_CONV1_SIZE, GEN_CONV1_SIZE],
        padding="same",
        activation=tf.nn.relu)

    GEN_CONV2_SIZE = 3
    gen_lay2_outs = tf.layers.conv2d(
        inputs=gen_lay1_outs,
        filters=POOLED_LAYER_SIZE,
        kernel_size=[GEN_CONV2_SIZE, GEN_CONV2_SIZE],
        padding="same",
        activation=tf.nn.relu)

    unpooled_genlay2 = unpool(gen_lay2_outs,DROPLAY_POOL_SIZE)

    skip_out_lay = tf.layers.conv2d(
        inputs=tot_droplay_outs,
        filters=LAY1_SIZE,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu)

    tot_unpooled = skip_out_lay + 0.1*unpooled_genlay2
    GEN_CONV3_SIZE = 3
    gen_lay3_outs = tf.layers.conv2d(
        inputs=tot_unpooled,
        filters=LAY1_SIZE,
        kernel_size=[GEN_CONV3_SIZE, GEN_CONV3_SIZE],
        padding="same",
        activation=tf.nn.relu)

    GEN_CONV4_SIZE = 3
    gen_lay4_outs = tf.layers.conv2d(
        inputs=gen_lay3_outs,
        filters=LAY1_SIZE,
        kernel_size=[GEN_CONV4_SIZE, GEN_CONV4_SIZE],
        padding="same",
        activation=tf.nn.relu)

    fin_outs = tf.layers.conv2d(
        inputs=gen_lay4_outs,
        filters=IMAGE_CHANNELS,
        kernel_size=[1, 1],
        padding="same",
        activation=None)
    fin_outs = tf.sigmoid(0.01*fin_outs)

    fin_out_lossess = sqr(in_img-fin_outs)
    #fin_out_lossess = tf.nn.sigmoid_cross_entropy_with_logits(labels=in_img,logits=fin_outs)
    flat_losses = tf.reshape(fin_out_lossess,(BATCH_SIZE, IMAGE_WIDTH*IMAGE_WIDTH*IMAGE_CHANNELS))
    batch_losses = tf.reduce_mean(flat_losses,axis=0)

    INFO_LAYER_SIZE = 16
    DROPLAY_POOL_SIZE = 2
    mask_drop_outs = tf.stop_gradient(tot_droplay_outs)
    pooled_outs = tf.layers.average_pooling2d(
        inputs=mask_drop_outs,
        pool_size=[DROPLAY_POOL_SIZE,DROPLAY_POOL_SIZE],
        strides=DROPLAY_POOL_SIZE,
    )

    MASK_CONV1_SIZE = 3
    mask_lay1_outs = tf.layers.conv2d(
        inputs=pooled_outs,
        filters=INFO_LAYER_SIZE,
        kernel_size=[MASK_CONV1_SIZE, MASK_CONV1_SIZE],
        padding="same",
        activation=tf.nn.relu)

    unpooled_masklay2 = unpool(mask_lay1_outs,DROPLAY_POOL_SIZE)

    skip_out_lay = tf.layers.conv2d(
        inputs=mask_drop_outs,
        filters=INFO_LAYER_SIZE,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu)

    tot_unpooled = skip_out_lay + unpooled_masklay2
    MASK_CONV3_SIZE = 3
    mask_lay3_outs = tf.layers.conv2d(
        inputs=tot_unpooled,
        filters=INFO_LAYER_SIZE,
        kernel_size=[MASK_CONV3_SIZE, MASK_CONV3_SIZE],
        padding="same",
        activation=tf.nn.relu)

    MASK_CONV4_SIZE = 3
    mask_lay4_outs = tf.layers.conv2d(
        inputs=mask_lay3_outs,
        filters=INFO_LAYER_SIZE,
        kernel_size=[MASK_CONV4_SIZE, MASK_CONV4_SIZE],
        padding="same",
        activation=tf.nn.relu)

    mask_lay5_outs = tf.layers.conv2d(
        inputs=mask_lay4_outs,
        filters=DROPOUT_CHANNELS,
        kernel_size=[1, 1],
        padding="same",
        activation=None)

    mask_outs = tf.sigmoid(0.01*mask_lay5_outs)

    return fin_outs, batch_losses, mask_outs

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

def get_sampled_indexes(mask_expectations,samples_per_item):
    all_samples = []
    mask_size = IMAGE_WIDTH*IMAGE_WIDTH*DROPOUT_CHANNELS
    for i in range(BATCH_SIZE):
        probabilities = norm_probs(UNIFORM_SAMPLE_PROB + mask_expectations[i].reshape(mask_size))
        next_choice = sample_probs(samples_per_item,probabilities)
        next_choice = np.minimum(next_choice,mask_size-1)
        all_samples.append(next_choice)
    return np.asarray(all_samples)

def set_sampled_indexes(old_mask,indexes):
    old_mask = np.copy(old_mask)
    all_samples = []
    mask_size = IMAGE_WIDTH*IMAGE_WIDTH*DROPOUT_CHANNELS
    for i in range(BATCH_SIZE):
        new_mask = old_mask[i].reshape((mask_size,))
        next_choice = indexes[i]
        new_mask[next_choice] = 1
        new_mask = new_mask.reshape((IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS))
        all_samples.append(new_mask)
    return np.stack(all_samples)

def calc_selection_masks(sampled_indicies, indici_losses):
    all_samp_masks = []
    all_max_masks = []
    mask_size = IMAGE_WIDTH*IMAGE_WIDTH*DROPOUT_CHANNELS
    for i in range(BATCH_SIZE):
        indicies = sampled_indicies[i]
        losses = indici_losses[i]
        samp_mask = np.zeros(mask_size)
        samp_mask[indicies] = 1

        max_loss_arg = np.argmax(losses)
        max_index = indicies[max_loss_arg]
        max_mask = np.zeros(mask_size)
        max_mask[max_index] = 1

        all_samp_masks.append(samp_mask)
        all_max_masks.append(max_mask)
    all_max_masks = np.stack(all_max_masks).reshape((BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS))
    all_samp_masks = np.stack(all_samp_masks).reshape((BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS))
    return all_max_masks, all_samp_masks

def save_image(float_data,filename):
    img_data = (float_data * 255.0).astype(np.uint8)
    img = Image.fromarray(img_data,mode="L")
    img.save(filename)

def html_to_int(code):
    return int("0x"+code[1::],16)*256+255

def save_mask(mask,fname):
    colors = [
        "#2E2EFE",
        "#00FF00",
        "#FF0040",
        "#610B0B",
        "#FF0040",
        "#00FFFF",
        "#FFFF00",
        "#848484",
        "#000000",
    ]
    color_ints = [html_to_int(col) for col in colors]
    resarray = np.zeros((IMAGE_WIDTH,IMAGE_WIDTH),dtype=np.int32)
    for x in range(IMAGE_WIDTH):
        for y in range(IMAGE_WIDTH):
            sum = 0
            iters = 0
            for z in range(DROPOUT_CHANNELS):
                if mask[x][y][z] != 0:
                    sum += color_ints[z]
                    iters += 1
            resarray[x][y] += sum // iters if iters != 0 else 0xffffffff

    casted_image = np.frombuffer(resarray.tobytes(), dtype=np.uint8).reshape((IMAGE_WIDTH,IMAGE_WIDTH,4))
    Image.fromarray(casted_image,mode="RGBA").save(fname)



def save_generated_image(orig_img,generated_image,generated_mask,revealed_capsules,weight_updates):
    folder = "generated/{}/".format(str(weight_updates))
    NUM_IMAGES_SAVE = 5
    if not os.path.exists(folder):
        for i in range(NUM_IMAGES_SAVE):
            subfold = folder+str(i)+"/"
            fname = subfold+"orig.png"
            os.makedirs(subfold)
            save_image(orig_img[i].reshape((IMAGE_WIDTH,IMAGE_WIDTH)),fname)
    for i in range(NUM_IMAGES_SAVE):
        fname = "{}{}/{}.png".format(folder,i,revealed_capsules)
        maskfname = "{}{}/{}m.png".format(folder,i,revealed_capsules)
        save_image(generated_image[i].reshape((IMAGE_WIDTH,IMAGE_WIDTH)),fname)
        save_mask(generated_mask[i],maskfname)

def is_power2(num):
	'states if a number is a power of two'
	return num != 0 and ((num & (num - 1)) == 0)

def learn_fn():
    in_img = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))
    dropout_mask = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS))

    mask_actual_max = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS)) # one for max selection 0 for everything else
    mask_actual_selection = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, DROPOUT_CHANNELS)) # one for selected vars, 0 for everything else

    #feed_input = tf.concat([in_img*mask,(mask)],axis=3)
    #AdamOptimizer
    #RMSPropOptimizer
    info_optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)
    #mask_optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)

    reconstruction, info_losses, generated_mask_expectations = mask_info(in_img, dropout_mask)
    tot_info_loss = tf.reduce_mean(info_losses)

    mask_losses = sqr(generated_mask_expectations-mask_actual_max)
    #mask_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask_actual_max,logits=generated_mask_expectations)
    mask_losses = mask_losses * mask_actual_selection
    mask_loss = tf.reduce_sum(mask_losses) / (BATCH_SIZE*SELECTION_SIZE)

    info_optim = info_optimizer.minimize(tot_info_loss)
    mask_optim = info_optimizer.minimize(mask_loss)

    #next_input = tf.stop_gradient(first_layer_out)

    train_data = np.copy(x_train)

    if os.path.exists("generated"):
        shutil.rmtree("generated")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #loss_val,opt_val = sess.run([loss,optim])
        update_num = 0
        for iteration_run in range(100000):
            np.random.shuffle(train_data)
            for x in range(0,len(train_data), BATCH_SIZE):
                update_num += 1
                data_sample = train_data[x:x+BATCH_SIZE].reshape((BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, IMAGE_CHANNELS))
                current_masks = init_mask()
                all_masks = [current_masks]
                # generate data
                for iter in range(150):
                    mask_expectations = sess.run(generated_mask_expectations,feed_dict={
                        in_img: data_sample,
                        dropout_mask: current_masks,
                    })
                    chosen_indexes = get_sampled_indexes(mask_expectations,1).reshape(BATCH_SIZE)
                    current_masks = set_sampled_indexes(current_masks,chosen_indexes)
                    all_masks.append(current_masks)
                    if is_power2(iter):
                        gen_img = sess.run(reconstruction,feed_dict={
                            in_img: data_sample,
                            dropout_mask: current_masks,
                        })
                        save_generated_image(data_sample,gen_img,current_masks,iter,update_num)

                print("generation finished")

                np_all_masks = np.concatenate(all_masks,axis=0)
                all_data = np.tile(data_sample,(len(all_masks),1,1,1))
                #train information optimizer
                tot_inf_loss = 0.0
                inf_iters = 0
                tot_mask_loss = 0.0
                for x in range(INFO_BUFFER_SIZE):
                    for li in range(3):
                        sample_idx = np.random.randint(0,len(np_all_masks),size=BATCH_SIZE)
                        loss_val,opt_val = sess.run([tot_info_loss,info_optim],feed_dict={
                            in_img: all_data[sample_idx],
                            dropout_mask: np_all_masks[sample_idx],
                        })
                        tot_inf_loss += loss_val
                        inf_iters += 1

                for x in range(INFO_BUFFER_SIZE):
                    sample_idx = np.random.randint(0,len(np_all_masks),size=BATCH_SIZE)

                    current_data = all_data[sample_idx]
                    current_mask = np_all_masks[sample_idx]
                    mask_expectations = sess.run(generated_mask_expectations,feed_dict={
                        in_img: current_data,
                        dropout_mask: current_mask,
                    })
                    sampled_indicies = get_sampled_indexes(mask_expectations,SELECTION_SIZE)
                    samp_indicies_t = np.transpose(sampled_indicies)
                    #print(samp_indicies_t)
                    all_samp_values = []
                    for samp in range(SELECTION_SIZE):
                        samp_values = sess.run(info_losses,feed_dict={
                            in_img: current_data,
                            dropout_mask: set_sampled_indexes(current_mask,samp_indicies_t[samp]),
                        })
                        all_samp_values.append(samp_values)
                    svals = np.stack(all_samp_values).transpose()

                    act_max, sel_mask = calc_selection_masks(sampled_indicies,svals)
                    loss_val,opt_val = sess.run([mask_loss,mask_optim],feed_dict={
                        in_img: all_data[sample_idx],
                        dropout_mask: np_all_masks[sample_idx],
                        mask_actual_max: act_max,
                        mask_actual_selection: sel_mask,
                    })

                    tot_mask_loss += loss_val

                print("info loss: {}".format(tot_inf_loss/inf_iters))
                print("mask loss: {}".format(tot_mask_loss/INFO_BUFFER_SIZE))


learn_fn()
