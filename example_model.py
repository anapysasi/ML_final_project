import tensorflow as tf
import numpy as np
from train_test_split import train_generator_func
from train_test_split import test_label_func
from sklearn.metrics import accuracy_score

train_generator = train_generator_func(info=True)
files, test_label, test_label_fruit = test_label_func()

image_classifier = tf.keras.Sequential()
image_classifier.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                            input_shape=train_generator.image_shape, activation='relu'))
image_classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
image_classifier.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
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

accuracy = accuracy_score(results, test_label)
print('The accuracy score is:', accuracy)