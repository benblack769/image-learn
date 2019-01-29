import scipy
import scipy.ndimage
import os
import numpy as np
import random
import sys
import tensorflow as tf
from PIL import Image
from make_gather_conv import make_gather_conv

BATCH_SIZE = 16

PIXEL_GEOM_P = 0.2
MAX_OFFSET = 40

IMAGE_SIZE = 256
GATHER_SIZE = 7

ADAM_learning_rate = 0.001

INPUT_CHANNELS = 6

OUT_COMPARE_SIZE = 16

def read_image(fname):
    data = scipy.ndimage.imread(fname)
    data = data.astype(np.float32)/255.0
    print(data.shape)
    return data


def save_rgb_image(img_data,filename):
    img_data = (img_data*255.0).astype(np.uint8)
    img = Image.fromarray(img_data,mode="RGB")
    img.save(filename)


def get_paths(folder):
    filenames = os.listdir(folder)
    jpg_filenames = [fname for fname in filenames if ".jpg" in fname]
    paths = [os.path.join(folder,fname) for fname in jpg_filenames]
    return paths

def get_images():
    folder = "example_images/"
    paths = get_paths(folder)
    images = [read_image(path) for path in paths]
    return images


def get_offsets():
    offsets = np.ones((BATCH_SIZE,2),dtype=np.int32) * 100000
    while np.any(offsets >= MAX_OFFSET):
        offsets = np.random.geometric(p=PIXEL_GEOM_P,size=(BATCH_SIZE,2))

    return offsets

def rand_image(all_images):
    return all_images[random.randrange(len(all_images))]

def filter_images(all_images):
    return [img for img in all_images
                if img.shape[0] > MAX_OFFSET + IMAGE_SIZE and
                   img.shape[1] > MAX_OFFSET + IMAGE_SIZE]

def randomly_crop_image(image, x_offset, y_offset):
    crop_height = image.shape[0] - y_offset - IMAGE_SIZE
    crop_width = image.shape[1] - x_offset - IMAGE_SIZE

    crop_pos_y = random.randrange(0,crop_height)
    crop_pos_x = random.randrange(0,crop_width)

    cropped_image = image[crop_pos_y:-(crop_height-crop_pos_y),
                          crop_pos_x:-(crop_width-crop_pos_x)]

    base_image = cropped_image[:-y_offset,:-x_offset]
    offset_image = cropped_image[y_offset:,x_offset:]

    return base_image,offset_image

def generate_offset_image_pairs_batch(filtered_images):
    batch_offsets = get_offsets()
    batch_cropped_images = []
    for i in range(BATCH_SIZE):
        x_off,y_off = batch_offsets[i]
        base_img,off_img = randomly_crop_image(rand_image(filtered_images), x_off, y_off)
        comb_img = np.concatenate([base_img,off_img],axis=2)
        batch_cropped_images.append(comb_img)
    return np.stack(batch_cropped_images), batch_offsets

def offset_cmp_vec(offsets):
    OFFSET_LAY1_size = 64
    #OFFSET_LAY2_size = 64
    one_hot_offsets = tf.one_hot(
        indices=offsets,
        depth=MAX_OFFSET,
        axis=-1,
        dtype=tf.float32
    )
    input_mat = tf.reshape(one_hot_offsets,(BATCH_SIZE*2,MAX_OFFSET*2))
    out1 = tf.layers.dense(input_mat,
                units=OFFSET_LAY1_size,
                activation=tf.nn.relu)
    #out2 = tf.layers.dense(out1,
    #            units=OFFSET_LAY2_size,
    #            activation=tf.nn.relu)
    out3 = tf.layers.dense(out1,
                units=OUT_COMPARE_SIZE,
                activation=None)
    return out3

def image_cmps(images):
    OUT_LAY1_SIZE = 128
    STRIDE_LEN = 4
    gather_fn = make_gather_conv(GATHER_SIZE,INPUT_CHANNELS)
    gather_fn_const = tf.constant(gather_fn)

    gathered_data = tf.nn.conv2d(images,
        filter=gather_fn_const,
        strides=(1,STRIDE_LEN,STRIDE_LEN,1),
        padding="VALID",
        )

    out2 = tf.layers.dense(gathered_data,
        units=OUT_LAY1_SIZE,
        activation=tf.nn.relu)

    out3 = tf.layers.dense(out2,
        units=OUT_LAY1_SIZE,
        activation=tf.nn.relu)

    out_cmp_vecs = tf.layers.dense(out2,
        units=OUT_COMPARE_SIZE,
        activation=None)

    return out_cmp_vecs

def get_loss(offset_cmps, img_cmps):
    offset_cmps = tf.reshape(offset_cmps,(2*BATCH_SIZE,1,OUT_COMPARE_SIZE))
    match_offset_vecs = offset_cmps[:BATCH_SIZE]
    mismatch_offset_vecs = offset_cmps[BATCH_SIZE:]

    img_shape = img_cmps.get_shape().as_list()
    vecs_per_img = img_shape[1] * img_shape[2]
    print(vecs_per_img)
    img_cmps = tf.reshape(img_cmps,(BATCH_SIZE,vecs_per_img,OUT_COMPARE_SIZE))

    match_guesses = tf.reduce_mean(match_offset_vecs * img_cmps,axis=2)
    mismatch_guesses = tf.reduce_mean(mismatch_offset_vecs * img_cmps,axis=2)
    all_guesses = tf.concat([match_guesses,mismatch_guesses],axis=0)

    result_val = tf.concat([tf.ones((BATCH_SIZE,vecs_per_img),dtype=tf.float32),
                            tf.zeros((BATCH_SIZE,vecs_per_img),dtype=tf.float32)],axis=0)

    all_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=result_val,logits=all_guesses)

    return tf.reduce_mean(all_losses)


def train_offset_pairs():
    filtered_imgs = filter_images(get_images())

    in_imgs = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, INPUT_CHANNELS))
    in_offsets = tf.placeholder(tf.int32, (2*BATCH_SIZE, 2))

    offset_cmps = offset_cmp_vec(in_offsets)
    img_cmps = image_cmps(in_imgs)

    loss = get_loss(offset_cmps, img_cmps)

    info_optimizer = tf.train.AdamOptimizer(learning_rate=ADAM_learning_rate)

    opt = info_optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000):
            loss_tot = 0
            for _ in range(100):
                imgs, match_offsets = generate_offset_image_pairs_batch(filtered_imgs)
                mismatch_offsets = get_offsets()
                all_offsets = np.concatenate([match_offsets,mismatch_offsets],axis=0)
                opt_val, loss_val = sess.run([opt,loss],feed_dict={
                    in_imgs:imgs,
                    in_offsets:all_offsets,
                })
                loss_tot += loss_val
            print(loss_tot/100)
            sys.stdout.flush()

train_offset_pairs()
#print(generate_offset_image_pairs_batch(filtered_imgs)[0].shape)
#get_offsets()
