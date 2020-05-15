#importing libraries 

import numpy as np 
import pandas as pd 
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns #Python data visualization library based on matplotlib.
pd.options.display.max_rows = None
pd.options.display.max_columns = None

#data preprocessing 

df = pd.read_csv("../input/bank_churn.csv")
feature_names = ['customer_id', 'credit_score','age','tenure','balance','products_number','credit_card','active_member','estimated_salary']
X = df[feature_names] #train
y = df["churn"] #test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling (very important)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#creating the artifical neural network
import keras 
import sys
from keras.models import Sequential #to initialize NN
from keras.layers import Dense #used to create layers in NN

#Initialising the ANN - Defining as a sequence of layers or a Graph
classifier = Sequential()

#Input Layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9 ))
#forward propagation 
#adding a hidden layer
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform'))
#adding another hidden layer
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform')) 

#adding the output layer
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
print("\n")
print(classifier)
print("prediction values")
y_pred = classifier.predict(X_test)

print(y_pred)
print("\n")
print("confusion matrix")
