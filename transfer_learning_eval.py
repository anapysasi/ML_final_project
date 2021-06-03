"""
This script loads an already built Image Recognition Transfer Learning model.
It uses the loaded InceptionV3 model to check how accurate it is on test data.
Accuracy here does not come nearly as close to the accuracy we see in the train,
validation phases due to the Batch Normalization error issue
that the InceptionV3 base model suffers from. (https://github.com/keras-team/keras/pull/9965)
"""
import tensorflow as tf
import numpy as np
import os
import warnings
import platform
from train_test_split import train_generator_func
from tensorflow.keras.models import load_model
from sklearn import metrics
from sklearn.metrics import accuracy_score
from train_test_split import test_label_func
warnings.filterwarnings('ignore')

if platform.system() == "Windows":
    train_dir = os.path.join('data\\Train')
    test_dir = os.path.join('data\\Test\\Groups')
else:
    train_dir = os.path.join('data/Train')
    test_dir = os.path.join('data/Test/Groups')

train_generator, train_datagen = train_generator_func(path=train_dir, target_size1=299,
                                                      target_size2=299, batch_size_val=16)
test_generator, test_datagen = train_generator_func(path=test_dir, target_size1=299,
                                                    target_size2=299, batch_size_val=16)

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
print('\nTest labels:', test_label)
print('\nPredicted labels:', results)

# Compute the accuracy score of the loaded model
accuracy = accuracy_score(test_label, results)
print('\n\nThe accuracy score is:', accuracy)

# Summarize the fit of the model on the test data
print('\n Classification report of the test data: \n')
print(metrics.classification_report(test_label, results))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(test_label, results, normalize='true').round(3))
