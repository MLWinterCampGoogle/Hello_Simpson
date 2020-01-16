from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import sys, os, cv2
from PIL import Image

import keras
from keras.preprocessing.image import load_img

# from IPython.display import clear_output
import matplotlib.pyplot as plt

from segmentation import unet_model, get_dataset

from load_cycle_gan_ltl import load_gan_model, image_pass_cycle_save


"""The following code performs a simple augmentation of flipping an image. In addition,  image is normalized to [0,1]. Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. For the sake of convenience, let's subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}."""

IMG_SIZE = 128

MODEL_PATH = sys.argv[1]
IMAGE_PATH = sys.argv[2]
CYCLE_GAN_PATH = sys.argv[3]

"""Let's take a look at an image example and it's correponding mask from the dataset."""

def display(display_list, output_path):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.savefig(output_path)

# for image, mask in train.take(1):
#   sample_image, sample_mask = image, mask
# display([sample_image, sample_mask])

def decode(image):
  img_tensor = tf.image.decode_image(image)
  img_final = tf.image.resize(img_tensor, [IMG_SIZE, IMG_SIZE])

  return img_final


"""Let's try out the model to see what it predicts before training."""

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def show_predictions(model, sample_image, dataset=None, num=1, epoch=0):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    # display([sample_image, sample_mask,
    #          create_mask(model.predict(sample_image[tf.newaxis, ...]))])
    prediction = model.predict(sample_image[tf.newaxis, ...])
    # print('prediction.shape', prediction.shape)
    # print('tf.shape(prediction)', tf.shape(prediction))
    output_tensor = create_mask(model.predict(sample_image[tf.newaxis, ...]))
    # print('output_tensor.shape:', output_tensor.shape)
  return output_tensor

def load_img(img_path):
  img_path_tensor = tf.io.read_file(img_path)
  img_tensor = decode(img_path_tensor)
  return img_tensor

def from_img_to_mask(model, img):
  
  output_mask = show_predictions(model, img)

  return output_mask

# out_sample_mask = tf.keras.preprocessing.image.array_to_img(sample_mask)
# out_sample_mask.save('Mattingl.png')

"""## Define the model
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py). 

The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.
# """

OUTPUT_CHANNELS = 2

"""As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in [tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process."""

# base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)

# # Use the activations of these layers
# layer_names = [
#     'block_1_expand_relu',   # 64x64
#     'block_3_expand_relu',   # 32x32
#     'block_6_expand_relu',   # 16x16
#     'block_13_expand_relu',  # 8x8
#     'block_16_project',      # 4x4
# ]
# layers = [base_model.get_layer(name).output for name in layer_names]

# # Create the feature extraction model
# down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

# down_stack.trainable = True

# """The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples."""

# up_stack = [
#     pix2pix.upsample(512, 3),  # 4x4 -> 8x8
#     pix2pix.upsample(256, 3),  # 8x8 -> 16x16
#     pix2pix.upsample(IMG_SIZE, 3),  # 16x16 -> 32x32
#     pix2pix.upsample(64, 3),   # 32x32 -> 64x64
# ]

# def unet_model(output_channels):

#   # This is the last layer of the model
#   last = tf.keras.layers.Conv2DTranspose(
#       output_channels, 3, strides=2,
#       padding='same', activation='softmax')  #64x64 -> IMG_SIZExIMG_SIZE

#   inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, 3])
#   x = inputs

#   # Downsampling through the model
#   skips = down_stack(x)
#   x = skips[-1]
#   skips = reversed(skips[:-1])

#   # Upsampling and establishing the skip connections
#   for up, skip in zip(up_stack, skips):
#     x = up(x)
#     concat = tf.keras.layers.Concatenate()
#     x = concat([x, skip])

#   x = last(x)

#   return tf.keras.Model(inputs=inputs, outputs=x)

# """## Train the model
# Now, all that is left to do is to compile and train the model. The loss being used here is losses.sparse_categorical_crossentropy. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and losses.sparse_categorical_crossentropy is the recommended loss for such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.
# """

# def load_test():
#   with open('.train_index.txt', 'r') as file_read:
#     file_read.
def get_dataset():
  with open('file_names.txt', 'r') as image_paths:
    all_image_paths = image_paths.readlines()
    all_image_paths = list(map(lambda x: x.strip('\n'), all_image_paths))


  with open('matting_names.txt', 'r') as matting_paths:
    all_matting_paths = matting_paths.readlines()
    all_matting_paths = list(map(lambda x: x.strip('\n'), all_matting_paths))

# TRAIN_LENGTH = info.splits['train'].num_examples

  with open('./train_index.txt', 'r') as file_r:
    train_index = file_r.read().splitlines()
    train_index = list(map(lambda x: int(x), train_index))


  path_ds = {}

  train_img_path = []
  train_mat_path = []

  test_img_path = []
  test_mat_path = []


  for i, path in enumerate(all_image_paths):
    try:
      image = load_img(all_image_paths[i])

      if i in train_index:
        train_img_path.append(all_image_paths[i])
        train_mat_path.append(all_matting_paths[i])
      else:
        test_img_path.append(all_image_paths[i])
        test_mat_path.append(all_matting_paths[i])
    except:
      pass

  print(len(train_img_path))
  print(len(test_img_path))

  with open('./train_index.txt', 'w+') as file_w:
    for index in train_index:
      file_w.write(str(index) + '\n')

  path_ds = {}
  path_ds['train'] = {}
  path_ds['test'] = {}

  path_ds['train']['image'] = tf.data.Dataset.from_tensor_slices(train_img_path)
  path_ds['train']['matting'] = tf.data.Dataset.from_tensor_slices(train_mat_path)

  path_ds['test']['image']= tf.data.Dataset.from_tensor_slices(test_img_path)
  path_ds['test']['matting']= tf.data.Dataset.from_tensor_slices(test_mat_path)

  # train = image_label_ds['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # test = image_label_ds['test'].map(load_image_test)

  # train_img = path_ds['train']['image'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # train_mat = path_ds['train']['matting'].map(load_mat_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  # test_img = path_ds['test']['image'].map(load_image_test)
  # test_mat = path_ds['test']['matting'].map(load_mat_test)


  # train_image_label_ds = tf.data.Dataset.zip((train_img, train_mat))
  # test_image_label_ds = tf.data.Dataset.zip((test_img, test_mat))
  # print(test_image_label_ds)


  # train_dataset = train_image_label_ds.take(STEPS_PER_EPOCH).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
  # train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  # test_dataset = test_image_label_ds.take(VALIDATION_STEPS).batch(BATCH_SIZE)

  return path_ds

def main():

  model = unet_model(OUTPUT_CHANNELS)

  model.load_weights(MODEL_PATH)
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  # checkpoint = True
  # if checkpoint:
  # 	model.load_weights(os.path.join(snapshots_dir, snapshots_file))
  # else:
  # 	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
  #               metrics=['accuracy'])

  """Have a quick look at the resulting model architecture:"""

  tf.keras.utils.plot_model(model, show_shapes=True)

  # datasets = get_dataset()
  # test_ds_image = datasets['test']['image']

  with open('./file_names.txt', 'r') as image_paths:
    all_image_paths = image_paths.readlines()
    all_image_paths = list(map(lambda x: x.strip('\n'), all_image_paths))

  cycle_gan_model = load_gan_model(CYCLE_GAN_PATH)


  for image_path in all_image_paths[:5]:

    image_tensor = load_img(image_path)
    # print(type(image_tensor))
    mask = from_img_to_mask(model, image_tensor)
    img_mask = tf.keras.preprocessing.image.array_to_img(mask)

    general_path = image_path.split('/')[-1]
    img_mask.save(os.path.join('./result/mask', general_path))

    img_BGR = cv2.imread(image_path)
    img_BGR = cv2.resize(img_BGR, (128, 128))
    origin_shape = img_BGR.shape

    cv_mask = cv2.imread(os.path.join('./result/mask', general_path), cv2.IMREAD_UNCHANGED)

    matted_img = cv2.bitwise_and(img_BGR, img_BGR, mask = cv_mask)
    predict_matting_path = os.path.join('./result/matting', general_path)
    cv2.imwrite(predict_matting_path, matted_img)




    input_img = Image.open(predict_matting_path)

    after_cycle_list = image_pass_cycle_save(cycle_gan_model, np.array(input_img))
    after_cycle_list = np.array(after_cycle_list)
    img_after_cycle_list = tf.keras.preprocessing.image.array_to_img(after_cycle_list)
    img_after_cycle_list.save(os.path.join('./result/processed', general_path))

    mask_inv = cv2.bitwise_not(cv_mask)
    matted_img = cv2.bitwise_and(img_after_cycle_list, img_after_cycle_list, mask = mask_inv)

    img_BGRA_inv = cv2.resize(cv2.merge((b_channel, g_channel, r_channel, alpha_inv)), (origin_shape[0], origin_shape[1]))

    # train_matting, test_matting = get_test_dataset()
    # matting_list = []
    # for mask in test_matting.take(5):
    # matting_list.append(mask)

    print('shape0', np.array(image_tensor).shape, 'shape1', after_cycle_list.shape)
    img_BGRA_predict = cv2.add(np.array(image_tensor),after_cycle_list)
    predict_whole_path = os.path.join('./result/whole', general_path.strip('.jpg') + '.png')
    img_BGRA_predict = cv2.cvtColor(img_BGRA_predict, cv2.COLOR_BGR2RGB)

    cv2.imwrite(predict_whole_path, img_BGRA_predict)





  # EPOCHS = 50
  # VAL_SUBSPLITS = 5


  # model_history = model.fit(train_dataset, epochs=EPOCHS,
  #                           steps_per_epoch=STEPS_PER_EPOCH,
  #                           # validation_steps=VALIDATION_STEPS,
  #                           # validation_data=test_dataset,
  #                           callbacks=[DisplayCallback(), saver_callback])


  # loss = model_history.history['loss']
  # # val_loss = model_history.history['val_loss']

  # epochs = range(EPOCHS)

  # plt.figure()
  # plt.plot(epochs, loss, 'r', label='Training loss')
  # # plt.plot(epochs, val_loss, 'bo', label='Validation loss')
  # plt.title('Training and Validation Loss')
  # plt.xlabel('Epoch')
  # plt.ylabel('Loss Value')
  # plt.ylim([0, 1])
  # plt.legend()
  # plt.savefig('board.jpg')

  # """## Make predictions

if __name__ == '__main__':

    main()