import os
import math
import random
import time
import numpy as np
import tensorflow as tf
import cv2
import threading
from PIL import Image
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

slim = tf.contrib.slim

IMAGE_SIZE_X = 300
IMAGE_SIZE_Y = 300


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.uint8, shape=(batch_size,IMAGE_SIZE_X*IMAGE_SIZE_Y*3))
    return images_placeholder

def fill_feed_dict(images_feed, images_pl):
    feed_dict = {
      images_pl: images_feed
    }
    return feed_dict
    
# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


test_images = []
path = '../demo/'
for filename in ['img220.jpg', 'img221.jpg', 'img222.jpg', 'img223.jpg']:
    image = Image.open(path+filename)
    image = image.resize((IMAGE_SIZE_X,IMAGE_SIZE_Y))
    test_images.append(np.array(image))

test_images = np.array(test_images)
test_images = test_images.reshape(4,IMAGE_SIZE_X*IMAGE_SIZE_Y*3)

images_placeholder=placeholder_inputs(4)


feed_dict = fill_feed_dict(test_images,images_placeholder)
isess.run([image_4d, predictions, localisations, bbox_img], feed_dict=feed_dict)

