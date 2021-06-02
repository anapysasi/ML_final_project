import tensorflow as tf
import numpy as np
from train_test_split import train_generator_func
from train_test_split import test_label_func

train_generator = train_generator_func(info=True, target_size1= 256, target_size2= 256)
files, test_label, test_label_fruit = test_label_func()

image_classifier = tf.keras.Sequential()
image_classifier.add(tf.keras.layers.Conv2D(16, kernel_size=3, padding="same", input_shape=(256, 256, 3)))
image_classifier.add(tf.keras.layers.LeakyReLU(alpha=0.2))
image_classifier.add(tf.keras.layers.Dropout(0.2))
image_classifier.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding="same"))
image_classifier.add(tf.keras.layers.LeakyReLU(alpha=0.2))
image_classifier.add(tf.keras.layers.Dropout(0.2))
image_classifier.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
image_classifier.add(tf.keras.layers.LeakyReLU(alpha=0.2))
image_classifier.add(tf.keras.layers.Dropout(0.2))
image_classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
image_classifier.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", input_shape=(256, 256, 3)))
image_classifier.add(tf.keras.layers.LeakyReLU(alpha=0.2))
image_classifier.add(tf.keras.layers.Dropout(0.2))
image_classifier.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding="same"))
image_classifier.add(tf.keras.layers.LeakyReLU(alpha=0.2))
image_classifier.add(tf.keras.layers.Dropout(0.2))
image_classifier.add(tf.keras.layers.Conv2D(512, kernel_size=3, padding="same"))
image_classifier.add(tf.keras.layers.LeakyReLU(alpha=0.2))
image_classifier.add(tf.keras.layers.Dropout(0.2))
image_classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
image_classifier.add(tf.keras.layers.Flatten())
image_classifier.add(tf.keras.layers.Dense(units=128, activation='relu'))
image_classifier.add(tf.keras.layers.Dense(units=4, activation='softmax'))
image_classifier.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

image_classifier.fit(train_generator, steps_per_epoch=3, epochs=3)

data = []
results = []
for f1 in files:
    img = tf.keras.preprocessing.image.load_img(f1, target_size=train_generator.image_shape)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    data.append(img)
    result = image_classifier.predict(img)
    r = np.argmax(result, axis=1)
    results.append(r)