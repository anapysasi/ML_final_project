"""
This script has a function [train_generator_func()] that creates a generator for the Train data. The path can be changed.
test_label_func() returns a vector with the files paths (files), the labels of the test files (test_label)
and a vector with specifically the  fruit (test_label_fruit)
"""
import os
import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import platform
import regex


def train_generator_func(path='data/Train', info=False, image=False, shear_range_val=0.2,
                         zoom_range_val=0.2, target_size1=68, target_size2=46, batch_size_val=32):
    """
    :param path: Path to retrieve the data.
    :param info: If True prints information about the generator.
    :param image: If True shows some of the images with the corresponding label
    :param shear_range_val: value for shear_range in ImageDataGenerator()
    :param zoom_range_val: value for zoom_range in ImageDataGenerator()
    :param target_size1: first value of the tuple for target_size in ImageDataGenerator()
    :param target_size2: second value of the tuple for target_size in ImageDataGenerator()
    :param batch_size_val: batch size in flow_from_directory()
    :return: train data generator
    """

    if platform.system() == 'Windows':
        if path == 'data/Train':
            path = 'data\\Train'
        else:
            pass
    else:
        pass

    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=shear_range_val,
                                                                           zoom_range=zoom_range_val,
                                                                           horizontal_flip=True)
    train_generator = train_data_generator.flow_from_directory(path, target_size=(target_size1, target_size2),
                                                               batch_size=batch_size_val, class_mode='categorical')
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
    return train_generator, train_data_generator


def test_label_func():
    """
    :return: returns a vector with the files paths (files), the labels of the test files (test_label)
             and a vector with specifically the  fruit (test_label_fruit)
    """
    if platform.system() == 'Windows':
        data_path = os.path.join('data\\Test', '*g')
    else:
        data_path = os.path.join('data/Test', '*g')

    files = glob.glob(data_path)

    test_label_fruit = list()
    test_label = list()
    label = None
    for f in files:
        if platform.system() == 'Windows':
            photo = regex.sub(r'data\\Test\\', '', f)
        else:
            photo = regex.sub(r'data/Test/', '', f)

        photo = regex.findall(r'[a-zA-Z]', photo)
        photo = ''.join(photo)
        photo = regex.sub(r'png', '', photo)
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
