import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn import metrics
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

print("hello")
BATCH_SIZE = 64
input_shape = (224, 224, 3)
n_classes=4

train_generator = ImageDataGenerator(rotation_range=90,
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5,
                                     height_shift_range=0.5,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input) # VGG16 preprocessing

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input) # VGG16 preprocessing

base_dir = Path('/Users/sj/Desktop/Things/UChicago/Winter 2021/ML_final_project')
# base_dir = '/Users/Sambhav Jain/PycharmProjects/ML_final_project'

train_data_dir = base_dir/'data/Train'
test_data_dir = base_dir/'data/Test/Groups'

class_subset = sorted(os.listdir(base_dir/'data/Train'))[:4] # Using only the first 4 classes


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


# vgg_model = create_model(input_shape, n_classes, fine_tune=0)
conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)

top_model = conv_base.output
top_model = Flatten(name="flatten")(top_model)
top_model = Dense(4096, activation='relu')(top_model)
top_model = Dense(1072, activation='relu')(top_model)
top_model = Dropout(0.2)(top_model)
output_layer = Dense(n_classes, activation='softmax')(top_model)

# Group the convolutional base and new fully-connected layers into a Model object.
vgg_model = Model(inputs=conv_base.input, outputs=output_layer)

# Generate predictions
vgg_model.load_weights('tl_model_v1.weights.best.hdf5') # initialize the best trained weights

true_classes = testgen.classes
class_indices = traingen.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

vgg_preds = vgg_model.predict(testgen)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)

vgg_acc = accuracy_score(true_classes, vgg_pred_classes)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))

print(true_classes)
print(vgg_pred_classes)

# Summarize the fit of the model on the test data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(true_classes, vgg_pred_classes))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(true_classes, vgg_pred_classes, normalize='true').round(3))