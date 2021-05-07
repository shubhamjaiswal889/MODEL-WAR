# -*- coding: utf-8 -*-

# -- Sheet --

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/data/workspace_files/healthcare-dataset-stroke-data.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
data = pd.read_csv("/data/workspace_files/healthcare-dataset-stroke-data.csv")
data.head()

data.isnull().sum()

data = data.fillna(0)

data.isnull().sum()

data.head()

Gender = pd.get_dummies(data['gender'], drop_first=True)
Ever_Married = pd.get_dummies(data['ever_married'], drop_first=True)
Work_Type = pd.get_dummies(data['work_type'], drop_first=True)
residence_type = pd.get_dummies(data['Residence_type'], drop_first=True)
Smoking_Status = pd.get_dummies(data['smoking_status'], drop_first=True)

data = pd.concat([data,Gender,Ever_Married, Work_Type, residence_type, Smoking_Status],axis=1)
data = data.drop(['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], axis = 'columns')

data.head()

X = data.drop('stroke', axis='columns')
y = data.stroke

X.head()

y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train.shape

X_test.shape

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import matplotlib.pyplot as plt

model = Sequential()
# add first hidden layer with input diamension
model.add(Dense(units = 32, activation='relu', kernel_initializer = 'he_uniform', input_dim = 17))
# add second hidden layer
model.add(Dense(units = 16, activation='relu', kernel_initializer = 'he_uniform'))
# add output layer
model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'glorot_uniform'))

# now we compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# train the model
model.fit(X_train, y_train, batch_size = 128, epochs = 50, verbose = 1)

acc = model.evaluate(X_test, y_test)

model.summary()

y_ann = model.predict(X_test)
y_ann = y_ann > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_ann, y_test)
cm

# Support Vector Machine


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model_svm = SVC()
model_svm.fit(X_train, y_train)
y_svm = model_svm.predict(X_test)
acc_svm = accuracy_score(y_svm, y_test)
cm_svm = confusion_matrix(y_svm, y_test)
acc_svm

cm_svm

from sklearn.metrics import classification_report
print(classification_report(y_svm, y_test))

# **Random ****Forest**


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_rf, y_test)
cm_rf = confusion_matrix(y_rf, y_test)
acc_rf

cm_rf

from sklearn.metrics import classification_report
print(classification_report(y_rf, y_test))

print("Artificial Neural Network Accuracy : ", acc)
print("Support Vector Machine Accuracy : ", acc_svm)
print("Random Forest Accuracy : ", acc_rf)



