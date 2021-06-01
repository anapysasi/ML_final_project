"""
This script assigns our train and test data to variables
"""

import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import regex


def train_generator_func(path='data/Train', info=False, image=False, shear_range_val=0.2,
                         zoom_range_val=0.2, target_size1=68, target_size2=46):
    """
    :param path: Path to retrieve the data.
    :param info: If True prints information about the generator.
    :param image: If True shows some of the images with the corresponding label
    :param shear_range_val: value for shear_range in ImageDataGenerator()
    :param zoom_range_val: value for zoom_range in ImageDataGenerator()
    :param target_size1: first value of the tuple for target_size in ImageDataGenerator()
    :param target_size2: second value of the tuple for target_size in ImageDataGenerator()
    :return: train data generator
    """

    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=shear_range_val,
                                                                           zoom_range=zoom_range_val,
                                                                           horizontal_flip=True)
    train_generator = train_data_generator.flow_from_directory(path, target_size=(target_size1, target_size2),
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

    test_label_fruit = list()
    test_label = list()
    label = None
    for f in files:
        photo = regex.sub(r"data/Test/", "", f)
        photo = regex.findall(r"[a-zA-Z]", photo)
        photo = ''.join(photo)
        photo = regex.sub(r"png", "", photo)
        if photo == 'Apple':
            label = 0
        if photo == 'Banana':
            label = 1
        if photo == 'Orange':
            label = 2
        if photo == 'Tamotoes':
            photo = 'Tomato'
            label = 3
        test_label_fruit.append(photo)
        test_label.append(label)
    return files, test_label, test_label_fruit
