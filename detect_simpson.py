# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:59:45 2020

@author: wangyi66
"""

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from data_loader import MattingImageGenerator
from keras.applications import vgg16
from keras.layers import Conv2D


# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

model_vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

graph = tf.get_default_graph()

image_input = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
layer3 = graph.get_tensor_by_name('layer3_out:0')
layer4 = graph.get_tensor_by_name('layer4_out:0')
layer7 = graph.get_tensor_by_name('layer7_out:0')

def added_layer(outputs):

	x = Flatten()(outputs)
	x = layers.Dense(4096, activation='relu', name='fc1')(x)
	x = layers.Dense(4096, activation='relu', name='fc2')(x)



# OUTPUTS = model_main.output


train_generator = MattingImageGenerator()

validation_generator = MattingImageGenerator()


# model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=800)
