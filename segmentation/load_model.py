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

    prediction = model.predict(sample_image[tf.newaxis, ...])

    output_tensor = create_mask(model.predict(sample_image[tf.newaxis, ...]))

  return output_tensor

def load_img(img_path):
  img_path_tensor = tf.io.read_file(img_path)
  img_tensor = decode(img_path_tensor)
  return img_tensor

def from_img_to_mask(model, img):
  
  output_mask = show_predictions(model, img)

  return output_mask

OUTPUT_CHANNELS = 2
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


  return path_ds

def main():

  model = unet_model(OUTPUT_CHANNELS)

  model.load_weights(MODEL_PATH)
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  """Have a quick look at the resulting model architecture:"""

  tf.keras.utils.plot_model(model, show_shapes=True)


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


if __name__ == '__main__':

    main()