import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl

import data
import module

def predict(img_name):
    load_size = 286
    crop_size = 256
    batch_size = 1
    A_img_paths_test = py.glob('photo', img_name)
    A_dataset_test = data.make_dataset(
            A_img_paths_test, batch_size, load_size, crop_size,
            training=False, drop_remainder=False, shuffle=False, repeat=1)

    G_A2B = module.ResnetGenerator(input_shape=(crop_size, crop_size, 3))
    G_B2A = module.ResnetGenerator(input_shape=(crop_size, crop_size, 3))
    
    tl.Checkpoint(dict(G_A2B=G_A2B, G_B2A=G_B2A), py.join('../CycleGAN-Tensorflow-2/output/simpson', 'checkpoints')).restore()
    
    @tf.function
    def sample_A2B(A):
        A2B = G_A2B(A, training=False)
        A2B2A = G_B2A(A2B, training=False)
        return A2B, A2B2A

    save_dir = 'result'
    py.mkdir(save_dir)
    for A in A_dataset_test:
        A2B, A2B2A = sample_A2B(A)
        for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
            img = np.concatenate([A_i.numpy(), A2B_i.numpy(), A2B2A_i.numpy()], axis=1)
            im.imwrite(img, py.join(save_dir, img_name.replace('img', 'jpg')))
