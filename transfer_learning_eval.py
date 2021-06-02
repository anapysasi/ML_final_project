"""
This script loads an already built Image Recognition Transfer Learning model.
It uses the loaded InceptionV3 model to check how accurate it is on test data.
Accuraccy here does not come nearly as close to the accuracy we see in the train, validation phases due to the Batch Normalization error issue
that the InceptionV3 base model suffers from. (https://github.com/keras-team/keras/pull/9965)
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from train_test_split import test_label_func
import os
import warnings
warnings.filterwarnings('ignore')

# Specify the base directory where images are located based on your machine & path
# Mac
base_dir = '/Users/sj/Desktop/Things/UChicago/Winter 2021/ML_final_project'
# Windows
# base_dir = '/Users/Sambhav Jain/PycharmProjects/ML_final_project'

# Specify the traning, validation, and test dirrectories.
train_dir = os.path.join(base_dir, 'data/Train')
test_dir = os.path.join(base_dir, 'data/Test/Groups')
augment_dir = os.path.join(base_dir, 'data/Augmentation')

# Normalize the pixels in the train data images, resize and augment the data.
train_datagen = ImageDataGenerator(
    rescale=1./255,# The image augmentaion function in Keras
    shear_range=0.2,
    zoom_range=0.2, # Zoom in on image by 20%
    horizontal_flip=True) # Flip image horizontally

augment_datagen = ImageDataGenerator(
    rescale=1./255,# The image augmentaion function in Keras
    shear_range=0.2,
    zoom_range=0.2, # Zoom in on image by 20%
    horizontal_flip=True) # Flip image horizontally

# Normalize the test data images, resize them but don't augment them
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')
augment_generator = augment_datagen.flow_from_directory(
    augment_dir,
    target_size=(299, 299),
    batch_size=16,
    class_mode='categorical')

# Check image shapes to see if they align
print(train_generator.image_shape)
print(test_generator.image_shape)
print(augment_generator.image_shape)

# Generate a test label to see what the actual data looks like
files, test_label, test_label_fruit = test_label_func()

# Load a pre-trained and fitted model
model = load_model("inceptionv3_transferlearning")
print('\n Model download successfully')

print('\n The final model is: \n', model)

data = []
results = []
for f1 in files:
    img = tf.keras.preprocessing.image.load_img(f1, target_size=train_generator.image_shape)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    data.append(img)
    result = model.predict(img)
    r = np.argmax(result, axis=1)
    results.append(r)

# Eyeball test for test_label and results
print(test_label)
print(results)

# Compute the accuracy score of the loaded model
accuracy = accuracy_score(test_label, results)
print('\n\nThe accuracy score is:', accuracy)

# Summarize the fit of the model on the test data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(test_label, results))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(test_label, results, normalize='true').round(3))
