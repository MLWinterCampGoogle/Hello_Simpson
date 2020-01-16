#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from numpy import load
from numpy import expand_dims
from keras.models import load_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot
 

# load an image to the preferred size
def preprocess(pixels):
	# # load and resize the image
	# pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = tf.image.resize(pixels, [256, 256])

	pixels = img_to_array(pixels)
	# transform in a sample
	pixels = expand_dims(pixels, 0)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	return pixels
 
# load the image
# image_src = load_image('horse2zebra/trainA/n02381460_541.jpg')
# load the model

def load_gan_model(cycle_gan_path):
	cust = {'InstanceNormalization': InstanceNormalization}
	model_AtoB = load_model(cycle_gan_path, cust, compile=False)
	return model_AtoB

def image_pass_cycle_save(model, image):
	image_src = preprocess(image)

	# translate image
	image_tar = model.predict(image_src)
	# scale from [-1,1] to [0,1]
	image_tar = (image_tar + 1) / 2.0
	# plot the translated image

	image_out = tf.image.resize(image_tar[0], [128, 128])
	return image_out



