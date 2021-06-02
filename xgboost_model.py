"""
Creates a more complex XGBoost model than the one in xgboost_simple_model.pu, by switching some of the parameters.
It uses `RandomizedSearchCV()` to decide what parameters fit the data best.
It can use either the original data or the augmented data depending on the parameter `augmentation`. The default is True.
"""
import xgboost as xgb
import pickle
import warnings
from sklearn import metrics
from sklearn.metrics import accuracy_score
from train_test_cv2 import train_generator_func
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings('ignore')

x_train, y_train, x_test, y_test = train_generator_func(augmentation=True)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples \n')

# Create XGB Classifier object
xgb_clf = xgb.XGBClassifier(tree_method="exact", predictor="cpu_predictor",
                            objective="multi:softmax")

parameters = {"learning_rate": [0.1, 0.01, 0.001],
              "gamma": [0.1, 0.5, 1, 2, 5],
              "max_depth": [2, 4, 6, 10],
              "subsample": [0.4, 0.6, 0.8, 1],
              "reg_alpha": [1, 1.5, 2, 3, 4.5, 10],
              "reg_lambda": [1, 2.5, 5, 7],
              "min_child_weight": [1, 3, 5, 7, 10],
              "n_estimators": [100, 250, 500, 800, 1000]}

# Create RandomizedSearchCV Object
xgb_rscv = RandomizedSearchCV(xgb_clf, param_distributions=parameters, scoring="f1_micro",
                              cv=5, verbose=3, random_state=123)

# Fit the model
xgb_rscv.fit(x_train, y_train, verbose=True)
print('\n\n Model best estimators:\n')
print("Learning Rate: ", xgb_rscv.best_estimator_.get_params()["learning_rate"])
print("Gamma: ", xgb_rscv.best_estimator_.get_params()["gamma"])
print("Max Depth: ", xgb_rscv.best_estimator_.get_params()["max_depth"])
print("Subsample: ", xgb_rscv.best_estimator_.get_params()["subsample"])
print("Alpha: ", xgb_rscv.best_estimator_.get_params()["reg_alpha"])
print("Lambda: ", xgb_rscv.best_estimator_.get_params()["reg_lambda"])
print("Minimum Sum of the Instance Weight Hessian to Make a Child: ",
      xgb_rscv.best_estimator_.get_params()["min_child_weight"])
print("Number of Trees: ", xgb_rscv.best_estimator_.get_params()["n_estimators"])

model = xgb_rscv.best_estimator_
model.fit(x_train, y_train)
print('\n The final model is: \n', model)

# save model to file
pickle.dump(model, open('xgboost_model2_augm.pickle.dat', 'wb'))

# Lets see how the model predicts on the train data
expected_train = y_train
predicted_train = model.predict(x_train)

# Summarize the fit of the model on the train data
print('\n Classification report of the train data: \n')
print(metrics.classification_report(expected_train, predicted_train))
print('\n Confusion matrix of the train data: \n')
print(metrics.confusion_matrix(expected_train, predicted_train, normalize='true').round(3))

# Make predictions
expected_y = y_test
predicted_y = model.predict(x_test)

# Summarize the fit of the model on the predictions
print('\n Classification report of the test data: \n')
print(metrics.classification_report(expected_y, predicted_y))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(expected_y, predicted_y, normalize='true').round(3))

# Lets calculate the accuracy:
accuracy = accuracy_score(expected_y, predicted_y)
print('\n\n The accuracy score is:', accuracy)
