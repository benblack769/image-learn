import scipy
import scipy.ndimage
import os
import numpy as np
import random
import tensorflow as tf
from PIL import Image

BATCH_SIZE = 4

PIXEL_GEOM_P = 0.1
MAX_OFFSET = 40

IMAGE_WIDTH = 256

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
    OFFSET_LAY1_size = 32
    OFFSET_LAY2_size = 64
    out1 = tf.layers.dense(offsets,
                units=OFFSET_LAY1_size,
                activation=tf.relu)
    out2 = tf.layers.dense(out1,
                units=OFFSET_LAY2_size,
                activation=tf.relu)
    out3 = tf.layers.dense(out2,
                units=OUT_COMPARE_SIZE,
                activation=None)
    return out3

def image_cmps(images):
    OUT_LAY1_SIZE = 64
    mask_lay1_outs = tf.layers.conv2d(
        inputs=images,
        filters=OUT_LAY1_SIZE,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    mask_lay2_outs = tf.layers.conv2d(
        inputs=mask_lay1_outs,
        filters=OUT_LAY1_SIZE,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    mask_lay3_outs = tf.layers.conv2d(
        inputs=mask_lay2_outs,
        filters=OUT_LAY1_SIZE,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu)
    mask_lay2_outs = tf.layers.conv2d(
        inputs=mask_lay3_outs,
        filters=OUT_COMPARE_SIZE,
        kernel_size=[1, 1],
        padding="same",
        activation=tf.nn.relu)


def train_offset_pairs():
    filtered_imgs = filter_images(get_images())

    in_imgs = tf.placeholder(tf.float32, (BATCH_SIZE, IMAGE_WIDTH, IMAGE_WIDTH, INPUT_CHANNELS))
    in_offsets = tf.placeholder(tf.float32, (2*BATCH_SIZE, 2))

    offset_cmps = offset_cmp_vec(in_offsets)
    img_cmps =

    with tf.Session() as sess:



print(generate_offset_image_pairs_batch(filtered_imgs)[0].shape)
#get_offsets()
