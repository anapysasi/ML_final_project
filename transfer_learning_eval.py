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

# Specify the base directory where images are located.
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

print(train_generator.image_shape)
print(test_generator.image_shape)
print(augment_generator.image_shape)

files, test_label, test_label_fruit = test_label_func()

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

print(test_label)
print(results)

accuracy = accuracy_score(test_label, results)
print('\n\nThe accuracy score is:', accuracy)

# Summarize the fit of the model on the train data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(test_label, results))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(test_label, results, normalize='true').round(3))


# # Lets see how the model predicts on the train data
# expected_train = y_train
# predicted_train = model.predict(x_train)
#
# # Summarize the fit of the model on the train data
# print('\n Classification report of the train data: \n')
# print(metrics.classification_report(expected_train, predicted_train))
# print('\n Confusion matrix of the train data: \n')
# print(metrics.confusion_matrix(expected_train, predicted_train, normalize='true').round(3))
#
# # Make predictions
# expected_y = y_test
# predicted_y = model.predict(x_test)
#
# # Summarize the fit of the model on the predictions
# print('\n Classification report of the test data: \n')
# print(metrics.classification_report(expected_y, predicted_y))
# print('\n Confusion matrix of the test data: \n')
# print(metrics.confusion_matrix(expected_y, predicted_y, normalize='true').round(3))
#
# # Lets calculate the accuracy:
# accuracy = accuracy_score(expected_y, predicted_y)
# print('\n\nThe accuracy score is:', accuracy)
