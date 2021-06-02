import xgboost as xgb
import pickle
import warnings
from sklearn import metrics
from sklearn.metrics import accuracy_score
from train_test_cv2 import train_generator_func
warnings.filterwarnings('ignore')

x_train, y_train, x_test, y_test = train_generator_func()
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples \n')

# fit a Xgboost model to the data
model = xgb.XGBClassifier()
model.fit(x_train, y_train, verbose=True)

# save model to file
pickle.dump(model, open("xgboost_simple_model.pickle.dat", "wb"))
print('\nModel saved successfully')

print('\n The final model is: ')
print(model)

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

# Summarize the fit of the model fro the predictions
print('\n Classification report of the test data: \n')
print(metrics.classification_report(expected_y, predicted_y))
print('\n Confusion matrix of the test data: \n')
print(metrics.confusion_matrix(expected_y, predicted_y, normalize='true').round(3))

# Lets calculate the accuracy:
accuracy = accuracy_score(expected_y, predicted_y)
print('\n\nThe accuracy score is:', accuracy)
