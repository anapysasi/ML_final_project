#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import matplotlib.pyplot as plt
import inspect
import os
import numpy as np
import matplotlib.pyplot as plt
from train_test_split import train_generator_func
from train_test_split import test_label_func
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend, models, layers, optimizers
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
import glob
import cv2
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
warnings.filterwarnings('ignore')


# In[42]:


from train_test_split import test_label_func
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


#Setting Directory
base_dir = '/Users/SahilSachdev/Downloads/MLPA-Final/'
train_dir = os.path.join(base_dir, 'data/Train')
validation_dir = os.path.join(base_dir, 'data/Test/test/Groups')


# In[3]:


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1.0/255)


# In[4]:


# Creating Train and Test Generators
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 32, class_mode = 'categorical', target_size = (224, 224))
validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size = 32, class_mode = 'categorical', target_size = (224, 224))


# In[5]:


print(train_generator.image_shape)


# In[6]:


def train_generator_func(img_size1=224, img_size_2=224):
    """

    :param img_size1: first value of the tuple to resize the image
    :param img_size_2: second value of the tuple to resize the image
    :return: x_train, y_train, x_test, y_test ready to use in xg_boost
    """
    train_path = glob.glob('data/Train/*/*')
    test_path = glob.glob('data/Test/test/Groups/*/*')

    def inp_process(file):
        data = []
        _label = None
        for f in file:
            try:
                part = f.split('/')
                assert part[-2] in ['Apple', 'Banana', 'Orange', 'Tomato']
                if part[-2] == 'Apple':
                    _label = 0
                elif part[-2] == 'Banana':
                    _label = 1
                elif part[-2] == 'Orange':
                    _label = 2
                elif part[-2] == 'Tomato':
                    _label = 3
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                r_siz = cv2.resize(img, (img_size1, img_size_2))
            except Exception as e:
                raise Exception(e)
            data.append([r_siz, _label])
        return np.array(data)

    train = inp_process(train_path)
    test = inp_process(test_path)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    return x_train, y_train, x_test, y_test


## Model 1 - Steps:100, Epochs:10 (ImageNet)

# In[7]:


from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D


base_model = ResNet50(input_shape=(224, 224,3), include_top=False, weights="imagenet")

for layer in base_model.layers:
    layer.trainable = False
    
base_model = Sequential()
base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(4, activation='sigmoid'))


# In[8]:


base_model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


# In[9]:


resnet_history = base_model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)


# In[10]:


base_model.save('ResNet50Base.h5')
print("Saved model")


# In[155]:


model = load_model("ResNet50Base.h5")
print('\n Model download successfully')


# In[156]:


from sklearn.metrics import accuracy_score

# Create a dictionary of the model history
history_dict = resnet_history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(history_dict['acc']) + 1)


# In[157]:


# Plot the training/validation loss
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[158]:


# Plot the training/validation accuracy
plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[159]:


# Evaluate the test accuracy and test loss of the model
test_loss, test_acc = base_model.evaluate_generator(validation_generator)
print('Model testing accuracy/testing loss:', test_acc, " ", test_loss)


# In[66]:


from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras import applications


# In[149]:


# Test Data Directory and Looping Over Images
test_img_dir = r"/Users/SahilSachdev/Downloads/MLPA-Final/data/Test/test/" 
data_path = os.path.join(test_img_dir, '*g')
files = glob.glob(data_path)

# Run prediction and add to results 
data = []
results = []

for f1 in files:
    img = image.load_img(f1, target_size = (225, 225))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    result = base_model.predict(img)
    r = np.argmax(result, axis=1)
    results.append(r)

print(results)


# In[160]:


true_classes = validation_generator.classes
class_indices = train_generator.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

resnet_preds = base_model.predict(validation_generator)
resnet_pred_classes = np.argmax(resnet_preds, axis=1)

resnet_acc = accuracy_score(true_classes, resnet_pred_classes)
print("ResNet50 Model Accuracy with Fine-Tuning: {:.2f}%".format(resnet_acc * 100))

print(true_classes)
print(resnet_pred_classes)

accuracy = accuracy_score(true_classes, resnet_pred_classes)
print('\n\nThe accuracy score is:', accuracy)

# Summarize the fit of the model on the test data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(true_classes, resnet_pred_classes))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(true_classes, resnet_pred_classes, normalize='true').round(3))


## Model 2 - Steps: 50; Epochs: 10 (None)

# In[14]:


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1.0/255)


# In[15]:


# Creating Train and Test Generators
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 32, class_mode = 'categorical', target_size = (224, 224))
validation_generator = test_datagen.flow_from_directory(validation_dir, batch_size = 32, class_mode = 'categorical', target_size = (224, 224))


# In[53]:


def train_generator_func(img_size1=224, img_size_2=224):
    """

    :param img_size1: first value of the tuple to resize the image
    :param img_size_2: second value of the tuple to resize the image
    :return: x_train, y_train, x_test, y_test ready to use in xg_boost
    """
    train_path = glob.glob('data/Train/')
    test_path = glob.glob('data/Test/test/Groups/')

    def inp_process(file):
        data = []
        _label = None
        for f in file:
            try:
                part = f.split('/')
                assert part[-2] in ['Apple', 'Banana', 'Orange', 'Tomato']
                if part[-2] == 'Apple':
                    _label = 0
                elif part[-2] == 'Banana':
                    _label = 1
                elif part[-2] == 'Orange':
                    _label = 2
                elif part[-2] == 'Tomato':
                    _label = 3
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                r_siz = cv2.resize(img, (img_size1, img_size_2))
            except Exception as e:
                raise Exception(e)
            data.append([r_siz, _label])
        return np.array(data)

    train = inp_process(train_path)
    test = inp_process(test_path)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    return x_train, y_train, x_test, y_test


# In[138]:


from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

base2_model = ResNet50(input_shape=(224, 224,3), include_top=False, weights=None)

for layer in base2_model.layers:
    layer.trainable = False
    
base2_model = Sequential()
base2_model.add(ResNet50(include_top=False, weights=None, pooling='max'))
base2_model.add(Dense(4, activation='sigmoid'))


# In[139]:


base2_model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


# In[140]:


resnet_history2 = base2_model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 50, epochs = 10)


# In[141]:


base2_model.save('ResNet50-Try2-2.h5')
print("Saved model")


# In[ ]:


resnet_history2 = load_model('ResNet50-Try2-2.h5')
print("Loaded model")


# In[143]:


from sklearn.metrics import accuracy_score

# Create a dictionary of the model history
history_dict2 = resnet_history2.history
loss_values2 = history_dict2['loss']
val_loss_values2 = history_dict2['val_loss']
acc_values2 = history_dict2['acc']
val_acc_values2 = history_dict2['val_acc']
epochs2 = range(1, len(history_dict2['acc']) + 1)


# In[144]:


# Plot the training/validation loss
plt.plot(epochs2, loss_values2, 'bo', label = 'Training loss')
plt.plot(epochs2, val_loss_values2, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[145]:


# Plot the training/validation accuracy
plt.plot(epochs2, acc_values2, 'bo', label = 'Training accuracy')
plt.plot(epochs2, val_acc_values2, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[146]:


# Evaluate the test accuracy and test loss of the model
test_loss2, test_acc2 = base2_model.evaluate_generator(validation_generator)
print('Model testing accuracy/testing loss:', test_acc2, " ", test_loss2)


# In[147]:


# Test Data Directory and Looping Over Images
test_img_dir = r"/Users/SahilSachdev/Downloads/MLPA-Final/data/Test/test/Groups/*" 
data_path = os.path.join(test_img_dir, '*g')
files = glob.glob(data_path)

# Run prediction and add to results 
data = []
results = []

for f1 in files:
    img = image.load_img(f1, target_size = (225, 225))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    result = base2_model.predict(img)
    r = np.argmax(result, axis=1)
    results.append(r)

print(results)


# In[153]:


true_classes = validation_generator.classes
class_indices = train_generator.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

resnet_preds = base2_model.predict(validation_generator)
resnet_pred_classes = np.argmax(resnet_preds, axis=1)

resnet_acc = accuracy_score(true_classes, resnet_pred_classes)
print("ResNet50 Model Accuracy without Fine-Tuning: {:.2f}%".format(resnet_acc * 100))

print(true_classes)
print(resnet_pred_classes)

accuracy = accuracy_score(true_classes, resnet_pred_classes)
print('\n\nThe accuracy score is:', accuracy)

# Summarize the fit of the model on the test data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(true_classes, resnet_pred_classes))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(true_classes, resnet_pred_classes, normalize='true').round(3))


## Model 3 - Steps:100; Epochs:20 (None)

# In[123]:


def train_generator_func(img_size1=225, img_size_2=225):
    """

    :param img_size1: first value of the tuple to resize the image
    :param img_size_2: second value of the tuple to resize the image
    :param augmentation: default True. Adds the images form the augmentation process to teh data.
    :param reshaped: default True. Reshapes the data
    :return: x_train, y_train, x_test, y_test ready to use in xg_boost
    """
    train_path = glob.glob('data/Train/*/*')
    test_path = glob.glob('data/Test/Groups/*/*')
   
    def inp_process(file):
        data = []
        _label = None
        for f in file:
            try:
                part = f.split('/')
                assert part[-2] in ['Apple', 'Banana', 'Orange', 'Tomato']
                if part[-2] == 'Apple':
                    _label = 0
                elif part[-2] == 'Banana':
                    _label = 1
                elif part[-2] == 'Orange':
                    _label = 2
                elif part[-2] == 'Tomato':
                    _label = 3
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                r_siz = cv2.resize(img, (img_size1, img_size_2))
            except Exception as e:
                raise Exception(e)
            data.append([r_siz, _label])
        return np.array(data)

    train = inp_process(train_path)
    test = inp_process(test_path)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    return x_train, y_train, x_test, y_test


# In[124]:


x_train, y_train, x_test, y_test = train_generator_func()


# In[128]:


from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout, Activation

base3_model = ResNet50(input_shape=(224, 224,3), include_top=False, weights=None)

for layer in base2_model.layers:
    layer.trainable = False
    
base3_model = Sequential()
base3_model.add(ResNet50(include_top=False, weights=None, pooling='max'))
base3_model.add(Dense(4, activation='sigmoid'))
base3_model.add(Activation("relu"))


# In[129]:


base3_model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


# In[130]:


resnet_history3 = base3_model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 50, epochs = 20)


# In[131]:


base3_model.save('ResNet50-Try3.h5')
print("Saved model")


# In[165]:


resnet_history3 = load_model('ResNet50-Try3.h5')
print("Loaded model")


# In[133]:


from sklearn.metrics import accuracy_score

# Create a dictionary of the model history
history_dict3 = resnet_history3.history
loss_values3 = history_dict3['loss']
val_loss_values3 = history_dict3['val_loss']
acc_values3 = history_dict3['acc']
val_acc_values3 = history_dict3['val_acc']
epochs3 = range(1, len(history_dict3['acc']) + 1)


# In[134]:


# Plot the training/validation loss
plt.plot(epochs3, loss_values3, 'bo', label = 'Training loss')
plt.plot(epochs3, val_loss_values3, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[135]:


# Plot the training/validation accuracy
plt.plot(epochs3, acc_values3, 'bo', label = 'Training accuracy')
plt.plot(epochs3, val_acc_values3, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[136]:


# Evaluate the test accuracy and test loss of the model
test_loss3, test_acc3 = base3_model.evaluate_generator(validation_generator)
print('Model testing accuracy/testing loss:', test_acc3, " ", test_loss3)


# In[166]:


# Test Data Directory and Looping Over Images
test_img_dir = r"/Users/SahilSachdev/Downloads/MLPA-Final/data/Test/test/Groups/*" 
data_path = os.path.join(test_img_dir, '*g')
files = glob.glob(data_path)

# Run prediction and add to results 
data = []
results = []

for f1 in files:
    img = image.load_img(f1, target_size = (225, 225))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = preprocess_input(img)
    result = base3_model.predict(img)
    r = np.argmax(result, axis=1)
    results.append(r)

print(results)


# In[142]:


true_classes = validation_generator.classes
class_indices = train_generator.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

resnet_preds = base3_model.predict(validation_generator)
resnet_pred_classes = np.argmax(resnet_preds, axis=1)

resnet_acc = accuracy_score(true_classes, resnet_pred_classes)
print("ResNet50 Model Accuracy without Fine-Tuning: {:.2f}%".format(resnet_acc * 100))

print(true_classes)
print(resnet_pred_classes)

accuracy = accuracy_score(true_classes, resnet_pred_classes)
print('\n\nThe accuracy score is:', accuracy)

# Summarize the fit of the model on the test data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(true_classes, resnet_pred_classes))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(true_classes, resnet_pred_classes, normalize='true').round(3))


# In[162]:


base3_model.summary()


# In[163]:


base2_model.summary()


# In[164]:


base_model.summary()

