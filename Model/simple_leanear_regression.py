import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import data set
dataset = pd.read_csv('../Dataset/GPS___Temp.csv')
# dataset['Time'] = pd.to_datetime(dataset['Time'],format= '%H:%M:%S' ).dt.time

X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values

# for set all decimal points to 5
len = X.__len__()

for x in range(len):
    X[x, 0] = round(X[x, 0], 5)
    X[x, 1] = round(X[x, 1], 5)

for z in range(len):
    temp = y[z].split(":")
    seconds = int(temp[0]) * 60 * 60 + int(temp[1]) * 60 + int(temp[2])
    y[z] = seconds
    print(seconds)

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
# X_test=sc_X.transform(X_test)


# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fitting simple linear regression model to training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_test_pred = regressor.predict(X_test)

# Visualizing the Training Set Results
# plt.scatter(X_train[:, 1], y_train, color='red')
# plt.plot(X_train[:, 1], y_train, color='yellow')
# plt.title('Time vs Long (Training Set)')
# plt.xlabel('Long')
# plt.ylabel('Time')
# plt.get_backend()
# plt.show()

# # Visualizing the Training Set Results
# plt.scatter(X_train[:, 2], y_train, color='red')
# plt.plot(X_train[:, 2], y_train, color='green')
# plt.title('Time vs Lat (Training Set)')
# plt.xlabel('Lat')
# plt.ylabel('Time')
# plt.get_backend()
# plt.show()
#
# # Visualizing the Test Set Results
# plt.scatter(X_test[:, 1], y_test, color='red')
# plt.plot(X_test[:, 1], y_test_pred, color='orange')
# plt.title('Time vs Long (Predicted Set)')
# plt.xlabel('Long')
# plt.ylabel('Time')
# plt.get_backend()
# plt.show()
#
# # Visualizing the Test Set Results
# plt.scatter(X_test[:, 2], y_test, color='red')
# plt.plot(X_test[:, 2], y_test_pred, color='pink')
# plt.title('Time vs Long (Predicted Set)')
# plt.xlabel('Lat')
# plt.ylabel('Time')
# plt.get_backend()
# plt.show()
