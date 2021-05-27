"""
This script assigns our train and test data to variables
"""

import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import re


def train_generator_func(info=False, image=False):
    """
    :param info: Prints information about the generator.
    :param image: Shows some of the images with the corresponding label
    :return: train data generator
    """

    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2,
                                                                           zoom_range=0.2, horizontal_flip=True)
    train_generator = train_data_generator.flow_from_directory('data/Train', target_size=(68, 46),
                                                               batch_size=32, class_mode='categorical')
    if info:
        print('The image shape of each training observation is:', train_generator.image_shape)
        print('\n We can see how:')
        for i in range(len(train_generator.class_indices)):
            print(list(train_generator.class_indices.keys())[i],
                  'is labeled as', list(train_generator.class_indices.values())[i])

    if image:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        for i, ax in enumerate(axs.flatten()):
            img, label = train_generator.next()
            _ = ax.set_title(f'Label = {np.argmax(label[i])}')
            _ = ax.imshow(img[i])
        plt.show()
    return train_generator


def test_label_func():
    """
    :return: returns a vector with the labels of the test files
    """
    data_path = os.path.join('data/Test', '*g')
    files = glob.glob(data_path)

    test_label = list()
    for f in files:
        photo = re.sub(r"data/Test/", "", f)
        photo = re.findall(r"[a-zA-Z]", photo)
        photo = ''.join(photo)
        photo = re.sub(r"png", "", photo)
        if photo == 'Tamotoes':
            photo = 'Tomato'
        test_label.append(photo)
    return test_label
