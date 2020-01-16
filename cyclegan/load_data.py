import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

import numpy as np
import os

def random_crop(image):
    cropped_image = tf.image.random_crop(
            image, size=[IMG_HEIGHT, IMG_WIDTH, 3])
    return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    return image

def random_jitter(image):
    image = tf.image.resize(image, [288, 288],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = random_crop(image)
    image = tf.image.random_flip_left_right(image)
    return image

def preprocess_image(image):
    image = random_jitter(image)
    image = normalize(image)
    return image

def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = preprocess_image(img)
    return img

def load_image_from_path(dir_path):
    dataset = tf.data.Dataset.list_files(dir_path)
    image_count = len(list(dataset))
    print(image_count)
    for f in dataset.take(5):
        print(f.numpy())
    return dataset

def load_paths(paths):
    dataset = None
    for path in paths:
        if dataset is None:
            dataset = load_image_from_path(path)
        else:
            dataset = dataset.concatenate(load_image_from_path(path))
    dataset = dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    for image in dataset.take(1):
        print("Image shape: ", image.numpy().shape)
    return dataset

def load_simpson_dataset():
    dataset = load_paths([#r'data/simpson440/cropped/*.png',
        r'data/simpson1k/simpsons_dataset/*/*.jpg'])
    return dataset

def load_human_dataset():
    dataset = load_paths([r'data/human/matting/*/*/*.png'])
    return dataset

if __name__ == '__main__':
    #load_simpson_dataset()
    load_human_dataset()
