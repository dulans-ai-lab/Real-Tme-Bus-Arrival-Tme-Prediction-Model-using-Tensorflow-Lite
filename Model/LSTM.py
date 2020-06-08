import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

# dataset = pd.read_csv('../Dataset/GPS__Temp.csv')
# dataset = pd.read_csv('../Dataset/GPS__Temp.csv',parse_dates=['Time'])
dataset = pd.read_csv('../Dataset/GPS__Temp.csv')
dataset['Time'] = pd.to_datetime(dataset['Time'],format= '%H:%M:%S' ).dt.time

X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 2:].values

# for set all decimal points to 4
len = X.__len__()

for y_range in range(len):
    y[y_range, 0] = round(y[y_range, 0], 4)
    y[y_range, 1] = round(y[y_range, 1], 4)


# Visualizing the Dataset
plt.scatter(y[:, 1], y[:, 0], color='red')
plt.plot(y[:, 1], y[:, 0], color='blue')
plt.title('Long vs Lat')
plt.xlabel('Latitude')
plt.ylabel('Longtide')
plt.get_backend()
plt.show()


# categorical data encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [1]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
X = transformer.fit_transform(X)

# Avoiding Dummy Variable Trap
X = X[:, 1:]

# feature scaling
from sklearn.preprocessing import StandardScaler
#
sc_X = StandardScaler()
y = sc_X.fit_transform(y)

# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


