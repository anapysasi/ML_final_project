import matplotlib.pyplot as plt
import warnings
import numpy as np
import tensorflow as tf
import glob
warnings.filterwarnings('ignore')

plot = True
save_img = False
it = None
items = ['Apple', 'Banana', 'Orange', 'Tomato']

for fruit in items:
    train_path = glob.glob('data/Train/' + fruit + '/*')
    augm_path = 'data/Augmentation/' + fruit + '/'

    # for f in train_path[0:int(0.2 * len(train_path))]:
    for f in train_path[3000:3004]:
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
