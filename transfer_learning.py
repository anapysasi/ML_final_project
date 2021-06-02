"""
This script builds an Image Recognition Transfer Learning model using tensorflow_hub. 
It builds and saves a InceptionV3 base model (>150 MB) with a few additional top layers.
Without a GPU, each epoch takes ~1.5 hours to run.
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from train_test_split import train_generator_func
from train_test_split import test_label_func
from sklearn.metrics import accuracy_score
from tensorflow.keras import backend, models, layers, optimizers
from tensorflow.python.client import device_lib
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from IPython.display import display
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
np.random.seed(32)


# Specify the base directory where images are located -  use the variable that corresponds to your local machine
# Mac
base_dir = '/Users/sj/Desktop/Things/UChicago/Winter 2021/ML_final_project'
# Windows
# base_dir = '/Users/Sambhav Jain/PycharmProjects/ML_final_project'

# Specify the traning, validation, and test dirrectories.
train_dir = os.path.join(base_dir, 'data/Train')
test_dir = os.path.join(base_dir, 'data/Test/Groups')

# Normalize the pixels in the train data images, resize and augment the data.
train_datagen = ImageDataGenerator(
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


# Always clear the backend before training a model
backend.clear_session()
# InceptionV3 model and use the weights from imagenet
conv_base = InceptionV3(weights = 'imagenet', #Useing the inception_v3 CNN that was trained on ImageNet data.
                  include_top = False)

# Connect the InceptionV3 output to the fully connected layers
InceptionV3_model = conv_base.output
pool = GlobalAveragePooling2D()(InceptionV3_model)
dense_1 = layers.Dense(512, activation = 'relu')(pool)
output = layers.Dense(4, activation = 'softmax')(dense_1)

# Create an example of the Archictecture to plot on a graph
model_example = models.Model(inputs=conv_base.input, outputs=output)
# plot graph
plot_model(model_example)

# Define/Create the model for training
model_InceptionV3 = models.Model(inputs=conv_base.input, outputs=output)
# Compile the model with categorical crossentropy for the loss function and SGD for the optimizer with the learning
# rate at 1e-4 and momentum at 0.9
model_InceptionV3.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
              metrics=['accuracy'])

# Import from tensorflow the module to read the GPU device and then print
print(device_lib.list_local_devices())

# Execute the model with fit_generator within the while loop utilizing the discovered GPU
history = model_InceptionV3.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience = 5, restore_best_weights = True)])

# Save model after running it so that you don't have to rerun it and can load it easily later
model_InceptionV3.save("inceptionv3_transferlearning_v2")

# Create a dictionary of the model history
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

# Plot the training/validation loss
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training/validation accuracy
plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the test accuracy and test loss of the model
test_loss, test_acc = model_InceptionV3.evaluate_generator(test_generator)
print('Model testing accuracy/testing loss:', test_acc, " ", test_loss)

