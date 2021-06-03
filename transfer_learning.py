"""
This script builds an Image Recognition Transfer Learning model using tensorflow_hub. 
It builds and saves a InceptionV3 base model (>150 MB) with a few additional top layers.
Without a GPU, each epoch takes ~1.5 hours to run.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import platform
from tensorflow.keras import backend, models, layers, optimizers
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from train_test_split import train_generator_func
np.random.seed(32)

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

# Always clear the backend before training a model
backend.clear_session()
# InceptionV3 model and use the weights from imagenet
# Using the inception_v3 CNN that was trained on ImageNet data.
conv_base = InceptionV3(weights='imagenet', include_top=False)

# Connect the InceptionV3 output to the fully connected layers
InceptionV3_model = conv_base.output
pool = GlobalAveragePooling2D()(InceptionV3_model)
dense_1 = layers.Dense(512, activation='relu')(pool)
output = layers.Dense(4, activation='softmax')(dense_1)

# Create an example of the Architecture to plot on a graph
model_example = models.Model(inputs=conv_base.input, outputs=output)
# plot graph
plot_model(model_example)

# Define/Create the model for training
model_InceptionV3 = models.Model(inputs=conv_base.input, outputs=output)
# Compile the model with categorical crossentropy for the loss function and SGD for the optimizer with the learning
# rate at 1e-4 and momentum at 0.9
model_InceptionV3.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                          metrics=['accuracy'])

# Import from tensorflow the module to read the GPU device and then print
print(device_lib.list_local_devices())

# Execute the model with fit_generator within the while loop utilizing the discovered GPU
history = model_InceptionV3.fit(
    train_generator, epochs=5, validation_data=test_generator, verbose=1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)])

# Save model after running it so that you don't have to rerun it and can load it easily later
model_InceptionV3.save("inceptionv3_transferlearning_v2")

# Create a dictionary of the model history
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

# Plot the training/validation loss
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training/validation accuracy
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the test accuracy and test loss of the model
test_loss, test_acc = model_InceptionV3.evaluate_generator(test_generator)
print('Model testing accuracy/testing loss:', test_acc, " ", test_loss)
