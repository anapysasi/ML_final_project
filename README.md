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

#### File: [`train_test_cv2.py`](https://github.com/anapysasi/ML_final_project/blob/main/train_test_cv2.py)

#### File: [`train_test_split.py`](https://github.com/anapysasi/ML_final_project/blob/main/train_test_split.py)

## XGBoost

#### File: [`xgboost_simple_model.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_simple_model.py)

#### File: [`xgboost_model.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model.py)

#### File: [`xgboost_eval.py`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_eval.py)

#### Model: [`xgboost_simple_model.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_simple_model.pickle.dat)

#### Model: [`xgboost_model.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model.pickle.dat)

#### Model: [`xgboost_model2.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model2.pickle.dat)

#### Model: [`xgboost_model2_augm.pickle.dat`](https://github.com/anapysasi/ML_final_project/blob/main/xgboost_model2_augm.pickle.dat)



File that prepares the 
