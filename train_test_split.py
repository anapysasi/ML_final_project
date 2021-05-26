# this script assigns our train and test data to variables

import tensorflow as tf
#import os
#import glob
#import numpy as np
#import pandas as pd
#import sklearn
import matplotlib as plt
#import warnings
#from sklearn.metrics import accuracy_score
from tensorflow import keras
#from keras.preprocessing.image import ImageDataGenerator
#from keras import Sequential
#from keras import layers
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import load_model
#warnings.filterwarnings('ignore')

train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_data_generator.flow_from_directory('data/Train', target_size=(38.4, 25.76),
                                                             batch_size=32, class_mode='categorical')
print('The image shape of each training observation is:', train_generator.image_shape)
print('The diferent clases are:', train_generator.class_indices, '\n\n Hence, we have 4 classes to predict on.')
#
# # Printing shape of data
img_shape = train_generator.image_shape
print(img_shape)
#
# # Visualizing images
fig, axs = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axs.flatten()):
      img, label = train_generator.next()
      _ = ax.set_title(f'Label = {np.argmax(label[i])}');
      _ = ax.imshow(img[i]);