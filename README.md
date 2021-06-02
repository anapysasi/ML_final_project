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

### File: [`image_augmentation.py`](https://github.com/anapysasi/ML_final_project/blob/main/image_augmentation.py)


## XGBoost

#### File: [`xgboost_simple_model.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_simple_model.py)

File that creates and saves a simple XGBoost model for the data. All the parameters are the default ones. The model is saves as [`xgboost_simple_model.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_simple_model.pickle.dat).

#### File: [`xgboost_model.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model.py)

Creates a more complex XGBoost model, by switching some of the parameters. It uses `RandomizedSearchCV()` to decide what parameters fit the data best. It can use either the original data or the augmented data depending on the parameter `augmentation`, teh dafault is `True`.

#### File: [`xgboost_eval.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_eval.py)

Evaluates the different XGBoost models. It opens one of the saved models (`xgboost_simple_model.pickle.dat`, `xgboost_model.pickle.dat`, `xgboost_model2.pickle.dat`, `xgboost_model2_aug.pickle.dat`) and fits the data (oiriganl or augmented). 

It retruns the Classification report, the Confusion matrix for the test and train data and the accuracy score of the model.

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

[[ 1. &emsp;  0.  &emsp; &nbsp;  0. &emsp;  &nbsp; 0.&emsp; &nbsp; ]<br>
 [ 0.51 &nbsp; 0.42 &nbsp; 0.06 &nbsp; 0.01 &nbsp; ]<br>
 [ 0.36 &nbsp; 0.07 &nbsp; 0.57 &nbsp; 0. &ensp; &nbsp; ]<br>
 [ 0.58 &nbsp; 0.03 &nbsp; 0.07 &nbsp; 0.32 ]]<br>

Accuracy score: 0.5775

#### Model: [`xgboost_model.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model.pickle.dat)

`RandomizedSearchCV()` with a dictionary of parameters. Best parameters:

- Learning Rate: 0.1
- Gamma:1 
- Max Depth:7 
- Subsample:0.7 
- Alpha:1 
- Lambda:3 
- Min Sum of Instance Weight to Make Child: 7 
- Number of Trees: 500

[[ 1. &emsp;  0.  &emsp; &nbsp;  0. &emsp;  &nbsp; 0.&emsp; &nbsp; ]<br>
 [ 0.44 &nbsp; 0.47 &nbsp; 0.07 &nbsp; 0.02 &nbsp; ]<br>
 [ 0.35 &nbsp; 0.06 &nbsp; 0.59 &nbsp; 0. &ensp; &nbsp; ]<br>
 [ 0.53 &nbsp; 0.03 &nbsp; 0.03 &nbsp; 0.41 ]]<br>
 
 Accuracy score: 0.6175

#### Model: [`xgboost_model2.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model2.pickle.dat)

`RandomizedSearchCV()` based on the results of the previous model.

- Learning Rate: 0.1
- Gamma:1 
- Max Depth:6 
- Subsample:0.6
- Alpha:1 
- Lambda:2.5
- Min Sum of Instance Weight to Make Child: 10 
- Number of Trees: 600

[[ 1. &emsp;  0.  &emsp; &nbsp;  0. &emsp;  &nbsp; 0.&emsp; &nbsp; ]<br>
 [ 0.41 &nbsp; 0.52 &nbsp; 0.06 &nbsp; 0.01 &nbsp; ]<br>
 [ 0.33 &nbsp; 0.03 &nbsp; 0.64 &nbsp; 0. &ensp; &nbsp; ]<br>
 [ 0.5 &ebsp; 0.04 &nbsp; 0.03 &nbsp; 0.43 ]]<br>
 
 Accuracy score: 0.6475

#### Model: [`xgboost_model2_augm.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model2_augm.pickle.dat)

RandomizedSearchCV() based on the results of the previous model + image augmentation

- Learning Rate: 0.1
- Gamma:1 
- Max Depth:6 
- Subsample:0.6
- Alpha:1 
- Lambda:2.5
- Min Sum of Instance Weight to Make Child: 10 
- Number of Trees: 800

[[ 1. &emsp;  0.  &emsp; &nbsp;  0. &emsp;  &nbsp; 0.&emsp; &nbsp; ]<br>
 [ 0.41 &nbsp; 0.56 &nbsp; 0.03 &nbsp; 0. &ebsp; &nbsp; ]<br>
 [ 0.29 &nbsp; 0.08 &nbsp; 0.63 &nbsp; 0. &ensp; &nbsp; ]<br>
 [ 0.49 &nbsp; 0.02 &nbsp; 0.03 &nbsp; 0.46 ]]<br>
 
 Accuracy score: 0.6625


