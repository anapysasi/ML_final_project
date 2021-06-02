# ML_final_project

### Authors: [Sambhav Jain](https://github.com/sambhavjain3211), [Ferando Forero](https://github.com/FernandoForeroAcosta), [Sahil Sachdev](https://github.com/sachdevsa) and Ana Ysasi.

Final project for Dr. Arnab Bose's Real Machine Learning & Predictive Analytics course at the University of Chicago's Master of Science in Analytics.

---

## Description:

In this project we aim to identify different fruits: apples, bananas, oranges and tomatoes through different Machine Learnign algorithms: CNN, XGBoost, ResNet50, and VGG16 transfer learning 

## Data:

Data retrieved from [kaggle: Fruit Recognition](https://www.kaggle.com/chrisfilo/fruit-recognition). We exported the Folders: `Apple`, `Banana`, `Orange` and `Tomato`. From this data, we also did a 20 % augmentation ([`image_augmentation.py`](https://github.com/anapysasi/ML_final_project/blob/main/image_augmentation.py)). In total, with the image augmentation, we had 15325 train samples images (12834 train samples originally) and 400 test samples.

---

## Installation Guide

```python
pip3 install gitpython

import os
from git.repo.base import Repo
Repo.clone_from("https://github.com/anapysasi/ML_final_project", "folderToSave")
```
---

## Quickstart Guide

## Handeling the data.

#### File: [`train_test_split.py`](https://github.com/anapysasi/ML_final_project/blob/main/train_test_split.py)

This script create a train generator and labels the test data. It has two functions:

  *  `train_generator_func()`: uses `ImageDataGenerator()` and `flow_from_directory` from keras to create a train generator. The path= of the data, the shear_range, zoom_range, target_size and batch_size can be customize. <br> If `info=True` return how many images are there in the gerator and from what classes. If `plot=True` plots 4 random images with its labels.
  *  `test_label_func()`: retunrs a vector with the files names(`files`), the labels of the classes for each corresponding file (`test_label`) and a vector with the corresponding fruit (`test_label_fruit`).

#### File: [`train_test_cv2.py`](https://github.com/anapysasi/ML_final_project/blob/main/train_test_cv2.py)

Has a function `train_generator_func()` that return the `x_train`, `y_train`, `x_test`, `y_test` from the dataset of images. Uses `cv2.IMREAD_GRAYSCALE` to read the images. You can customize the resize of the images (by changing the variables `img_size1`, `img_size_2`). The default size is (68,46).

The default of the function is to __reshape__ the data and __normailize__ it with `StandardScaler()`. This functionality can be trun of by setting `reshaped=False` as an argument.

By default it takes all the images available, including the ones form the data augmentation. If you set `augmentation=False` it would only use the origianl images to train the model.


## XGBoost

#### File: [`xgboost_simple_model.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_simple_model.py)

File that creates and saves a simple XGBoost model for the data. All the parameters are the default ones.

#### File: [`xgboost_model.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model.py)

#### File: [`xgboost_eval.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_eval.py)

#### Model: [`xgboost_simple_model.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_simple_model.pickle.dat)

Run with the default variables. No changes to the function. 

- Learning Rate: 0.3
- Gamma: 0
- Max Depth: 6
- Subsample: 1
- Alpha: 0
- Lambda: 1
- Min Sum of Instance Weight to Make Child: 1
- Number of Trees: 100

[[ 1. &emsp;  0.  &emsp;  0. &emsp;  0.]<br>
 [ 0.51 &nbsp; 0.42 &nbsp; 0.06 &nbsp; 0.01 ]<br>
 [ 0.36 &nbsp; 0.07 &nbsp; 0.57 &nbsp; 0. &ensp; &nbsp; ]<br>
 [ 0.58 &nbsp; 0.03 &nbsp; 0.07 &nbsp; 0.32 ]]<br>

accuracy: 0.5775

#### Model: [`xgboost_model.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model.pickle.dat)

#### Model: [`xgboost_model2.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model2.pickle.dat)

#### Model: [`xgboost_model2_augm.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model2_augm.pickle.dat)


