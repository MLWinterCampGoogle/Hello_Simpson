#!/usr/bin/env python
# coding: utf-8

# In[18]:


from numpy import load
from matplotlib import pyplot
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import load_model
from numpy import savez_compressed
from numpy import load
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


#from tensorflow.compat.v1.keras.preprocessing.image import load_img
#tf.disable_v2_behavior()
def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

A_data, B_data = load_real_samples('people_256.npz')
print('Loaded', A_data.shape, B_data.shape)

cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_035616.h5', cust)
model_BtoA = load_model('g_model_BtoA_035616.h5', cust)

def select_sample(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    return X

A_real = select_sample(A_data, 1)
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)

def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()
    
#filename = 'Hello.png' 
#pyplot.savefig(filename)    


show_plot(A_real, B_generated, A_reconstructed)


# In[ ]:





# In[ ]:




