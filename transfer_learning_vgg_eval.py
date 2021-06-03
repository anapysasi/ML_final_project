"""
This script loads an already built Image Recognition Transfer Learning model.
It uses the loaded VGG16 model to check how accurate it is on test data.
VGG16 does not use any Batch Normalization, so it does not run into any issues.
"""
import os
import numpy as np
import platform
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

# Set parameters used in creating the train, test, validation generators
BATCH_SIZE = 64
input_shape = (224, 224, 3)
n_classes = 4

train_generator = ImageDataGenerator(rotation_range=90,
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5,
                                     height_shift_range=0.5,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

if platform.system() == "Windows":
    train_data_dir = os.path.join('data\\Train')
    test_data_dir = os.path.join('data\\Test\\Groups')
else:
    train_data_dir = os.path.join('data/Train')
    test_data_dir = os.path.join('data/Test/Groups')

class_subset = sorted(train_data_dir)[:4]

traingen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='training',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)

validgen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=class_subset,
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)

testgen = test_generator.flow_from_directory(test_data_dir,
                                             target_size=(224, 224),
                                             class_mode=None,
                                             classes=class_subset,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=42)

# Create an instance of a VGG16 model that you can load the pre-saved weights to, without re-fitting the model
conv_base = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

# Replicating the exact model on which the weights are saved
top_model = conv_base.output
top_model = Flatten(name="flatten")(top_model)
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dense(1072, activation='relu')(top_model)
top_model = Dropout(0.2)(top_model)
output_layer = Dense(n_classes, activation='softmax')(top_model)

# Group the convolutional base and new fully-connected layers into a Model object.
vgg_model = Model(inputs=conv_base.input, outputs=output_layer)

# Generate predictions
# initialize the best trained weights
vgg_model.load_weights('tl_model_v1.weights.best.hdf5')

true_classes = testgen.classes
class_indices = traingen.class_indices
class_indices = dict((v, k) for k, v in class_indices.items())

vgg_preds = vgg_model.predict(testgen)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)

vgg_acc = accuracy_score(true_classes, vgg_pred_classes)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))

# Eyeball test to see how our model is doing
print(true_classes)
print(vgg_pred_classes)

accuracy = accuracy_score(true_classes, vgg_pred_classes)
print('\n\nThe accuracy score is:', accuracy)

# Summarize the fit of the model on the test data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(true_classes, vgg_pred_classes))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(true_classes, vgg_pred_classes, normalize='true').round(3))
