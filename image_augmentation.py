"""
Function that makes the image augmentation of the data.
By default it saves the new images in a folder called "Augmentation". The destination path can be changed.
In the destination there needs to be four folders called: Apple, Banana, Orange, Tomato
It makes the following transformations to the data:
- Horizontal and Vertical Shift Augmentation
- Random Rotation Augmentation
- Random Brightness Augmentation
- Random Zoom Augmentation
"""
import matplotlib.pyplot as plt
import warnings
import numpy as np
import tensorflow as tf
import glob
import platform
warnings.filterwarnings('ignore')


def img_augmentation(save_img=True, plot=False, augmentation_path='data/Augmentation/', percentage=0.2):
    """
    :param save_img: Default true. Saves the images to the augmentation_path. The path needs to end in / (if Windows \\)
    :param plot: Default False. If True it shows an example of the transformation.
    :param augmentation_path: Only needed if save_img=True. Path where the images are going to be saved.
    :param percentage: Default 0.2. Number from (0,1] that corresponds to the percentage of images we want to create.
    :return:
    """
    it = None
    items = ['Apple', 'Banana', 'Orange', 'Tomato']

    for fruit in items:
        if platform.system() == "Windows":
            train_path = glob.glob('data\\Train\\' + fruit + '\\*')
            if augmentation_path == 'data/Augmentation/':
                augm_path = 'data\\Augmentation\\' + fruit + '\\'
            else:
                augm_path = augmentation_path + fruit + '\\'
        else:
            train_path = glob.glob('data/Train/' + fruit + '/*')
            augm_path = augmentation_path + fruit + '/'

        for f in train_path[0:int(percentage * len(train_path))]:
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=[-50, 50],
                                                                      height_shift_range=0.4, rotation_range=90,
                                                                      brightness_range=[0.2, 1.0],
                                                                      zoom_range=[0.5, 1.0])

            img = tf.keras.preprocessing.image.load_img(f)
            data = tf.keras.preprocessing.image.img_to_array(img)
            samples = np.expand_dims(data, 0)
            if save_img:
                it = datagen.flow(samples, batch_size=1,
                                  save_to_dir=augm_path,
                                  save_prefix=fruit + '_augm',
                                  save_format="png")
            else:
                it = datagen.flow(samples)
            batch = it.next()
            image = batch[0].astype('uint8')

    if plot:
        for i in range(9):
            plt.subplot(330 + 1 + i)
            batch = it.next()
            image = batch[0].astype('uint8')
            plt.imshow(image)
        plt.show()
