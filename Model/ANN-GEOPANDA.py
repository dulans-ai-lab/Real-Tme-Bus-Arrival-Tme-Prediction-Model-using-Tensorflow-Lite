import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as geopd


# import data set
dataset = pd.read_csv('../Dataset/GPS___Temp1.csv')
# dataset = geopd.read_file('../Dataset/GPS___Temp.csv')
dataset['Time'] = pd.to_datetime(dataset['Time'],format= '%H:%M:%S' ).dt.time

# dataset.plot()
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values
#
# # for set all decimal points to 5
# len = dataset.geometry.length.__len__()
#
#
# for x in range(len):
#     X[x, 0] = round(X[x, 0], 5)
#     X[x, 1] = round(X[x, 1], 5)
#
# for z in range(len):
#     temp = y[z].split(":")
#     seconds = int(temp[0]) * 60 * 60 + int(temp[1]) * 60 + int(temp[2])
#     y[z] = seconds
#     print(seconds)

# categorical data encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [2]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
X = transformer.fit_transform(X)

# Avoiding Dummy Variable Trap
X = X[:, 1:]

# feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create ANN Model and fit into dataset
import keras
from keras.models import Sequential
from keras.layers import Dense

# #     Initializing ANN
# classifier_ann = Sequential()
#
# #    Adding the input layer and first hidden layer
# classifier_ann.add(Dense(units=3, kernel_initializer='uniform', activation='relu'))
#
# #    Adding the second hidden layer
# classifier_ann.add(Dense(units=2, kernel_initializer='uniform', activation='relu'))
#
# #    Adding the output layer
# classifier_ann.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))
#
# #    Compiling ANN
# classifier_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# #   Fitting the ANN to Training Set
# classifier_ann.fit(X_train, y_train, batch_size=10, epochs=100)

# Predict using ANN Model
# y_pred = classifier_ann.predict(X_test)

# # converting probabilities to prediction results
# _pred=(y_pred>0.5)y

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
#
# cm = confusion_matrix(y_test, y_pred)


# Create ANN Regression Model

from sklearn.neural_network import MLPRegressor

neural_network = MLPRegressor(hidden_layer_sizes=(200, 200), activation="logistic", max_iter=50000, solver="lbfgs")

neural_network.fit(X_train, y_train)
y_pred = neural_network.predict(X_test)
