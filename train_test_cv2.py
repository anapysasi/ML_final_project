"""
Returns the x_train, y_train, x_test and y_test of the data.
With all the default parameters the train_generator_func() creates the right input for the XGBoost algorithm.
It uses cv2.IMREAD_GRAYSCALE to open the images.
It resizes the images by default to 68x46.
By default it reshapes the images and normalizes them. This can be turned off.
"""
import numpy as np
import glob
import cv2
import warnings
import platform
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


def train_generator_func(img_size1=68, img_size_2=46, augmentation=True, reshaped=True):
    """

    :param img_size1: first value of the tuple to resize the image
    :param img_size_2: second value of the tuple to resize the image
    :param augmentation: default True. Adds the images form the augmentation process to the data.
    :param reshaped: default True. Reshapes the data
    :return: x_train, y_train, x_test, y_test ready to use in xg_boost
    """
    augm_path = None
    if platform.system() == "Windows":
        train_path = glob.glob('data\\Train\\*\\*')
        test_path = glob.glob('data\\Test\\Groups\\*\\*')
        if augmentation:
            augm_path = glob.glob('data\\Augmentation\\*\\*')
    else:
        train_path = glob.glob('data/Train/*/*')
        test_path = glob.glob('data/Test/Groups/*/*')
        if augmentation:
            augm_path = glob.glob('data/Augmentation/*/*')

    def inp_process(file):
        data = []
        _label = None
        for f in file:
            try:
                part = f.split('/')
                assert part[-2] in ['Apple', 'Banana', 'Orange', 'Tomato']
                if part[-2] == 'Apple':
                    _label = 0
                elif part[-2] == 'Banana':
                    _label = 1
                elif part[-2] == 'Orange':
                    _label = 2
                elif part[-2] == 'Tomato':
                    _label = 3
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                r_siz = cv2.resize(img, (img_size1, img_size_2))
            except Exception as e:
                raise Exception(e)
            data.append([r_siz, _label])
        return np.array(data)

    train = inp_process(train_path)
    test = inp_process(test_path)
    augm = None
    if augmentation:
        augm = inp_process(augm_path)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)

    if augmentation is True:
        for feature, label in augm:
            x_train.append(feature)
            y_train.append(label)

    if reshaped:
        # x_train (original) is 12834 rows of (68x46) values --> reshaped in 12834 x 3128
        # x_train (augmented) is 15325 rows of (68x46) values --> reshaped in 15325 x 3128
        # y_train is 400 rows of (68x46) values --> reshaped in 400 x 3128
        reshaped_size = img_size1 * img_size_2

        if not augmentation:
            x_train = np.array(x_train).reshape(12834, reshaped_size)
        else:
            x_train = np.array(x_train).reshape(15325, reshaped_size)
        x_test = np.array(x_test).reshape(400, reshaped_size)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Rescale the values to a smaller range with mean zero and unit variance.
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
    return x_train, y_train, x_test, y_test
