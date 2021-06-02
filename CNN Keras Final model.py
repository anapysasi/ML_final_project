import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import glob
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# PLEASE RUN WITH TENSORFLOW 2.1.0 VERSION

# Creating image data generation instance
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255, horizontal_flip = True)

# Creating a train generator using prvious instance
X_train = train_datagen.flow_from_directory(
'C:\\Users\\fmfor\\Desktop\\Train', target_size = (64, 64), batch_size = 32, class_mode = 'categorical')

# Printing shape of data
img_shape = X_train.next()[0][0].shape

# Creating a classifier
def build_class():
    classifier = tf.keras.Sequential()
    classifier.add(
        tf.keras.layers.Conv2D(
            16, kernel_size=(3, 3),
            activation='relu',
            input_shape=img_shape))
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    classifier.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    classifier.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    classifier.add(tf.keras.layers.Flatten())
    classifier.add(tf.keras.layers.Dense(128, activation='relu'))
    classifier.add(tf.keras.layers.Dense(4, activation='softmax'))

    classifier.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy'])
    return classifier

# Extracting and visualizing test images
files = glob.glob(os.path.join('C:\\Users\\fmfor\\Desktop\\Test', '*.png'))

X_test = []
#fig, axs = plt.subplots(400, 1, figsize=(16, 16));
y_test = []
for (i, file) in enumerate(files):
    img = image.load_img(file, target_size = (64, 64))
    img = image.img_to_array(img);
    img = np.expand_dims(img, axis = 0)
    X_test.append(img)

# Creating basline model and saving it on disk
try:
    classifier = load_model('C:\\Users\\fmfor\\Downloads\\b_model-new5.h5')
    print('Baseline model loaded from disk')
except:
    classifier = build_class()
    steps_per_epoch = 3
    epochs = 3
    classifier.fit(
        X_train,
        steps_per_epoch=X_train.samples // 32 +
        min(X_train.samples % 32, 1),
        epochs=epochs)
    classifier.save('C:\\Users\\fmfor\\Downloads\\b_model-new5.h5')
    print('Baseline model saved')

# Creating function for predictions
def predict(model):
    y_pred = []
    for i in X_test:
        y_pred.append(np.argmax(model.predict(i)))
    return y_pred, accuracy_score(y_test, y_pred)

# Loading models
models = {}

for j in [3,4]:
    for i in [2,3,4]:
        try:
            models[f'{i}_{j}'] = load_model(f'./Models/model_{i}_{j}.h5')
            print(f'Model_{i}_{j} loaded')
        except:
            classifier = build_class()
            classifier.fit(
                train_datagen.flow_from_directory(
                    'C:\\Users\\fmfor\\Desktop\\Train',
                    target_size=(64, 64),
                    batch_size=X_train.samples // i +
                    min(X_train.samples % i, 1),
                    class_mode='categorical'),
                epochs=j,
                verbose=0)
            models[f'{i}_{j}'] = classifier
            classifier.save(f'./Models/model_{i}_{j}.h5')
            print(f'Model_{i}_{j} saved')

# Printing results sorted by accuracy score
results = pd.DataFrame({
    'Step #':[eval(x.replace('_', ','))[0] for x in models],
    'Steps per Epoch #':[eval(x.replace('_', ','))[1] for x in models],
    'Accuracy Score':[predict(models[x])[1] for x in models]
})

results.sort_values(by=['Accuracy Score'])

# Printing confusion matrix

confusion_matrix(y_test, predict(classifier)[0])