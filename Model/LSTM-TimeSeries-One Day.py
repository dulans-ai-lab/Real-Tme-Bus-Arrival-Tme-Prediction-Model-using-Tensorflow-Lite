import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('../Dataset/GPS Database Cleaned Data-One Day.csv', parse_dates=True, index_col='date_time')

# for set all decimal points to 4
dataset = np.array(dataset)
len = dataset.shape[0]

for row in range(len):
    dataset[row, 8] = round(dataset[row, 8], 5)
    dataset[row, 9] = round(dataset[row, 9], 5)
    dataset[row, 10] = round(dataset[row, 10], 5)

# categorical data encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [0]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
dataset = transformer.fit_transform(dataset)

# Avoiding Dummy Variable Trap
# dataset = dataset[:, 1:]

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [1]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
dataset = transformer.fit_transform(dataset)

# Avoiding Dummy Variable Trap
# dataset = dataset[:, 1:]

transformer = ColumnTransformer(
    transformers=[
        ("OneHotEncoder",  # Just a name
         OneHotEncoder(),  # The transformer class
         [2]  # The column(s) to be applied on.
         )
    ],
    remainder='passthrough'
)
dataset = transformer.fit_transform(dataset)


dataset=dataset.astype('float32')

# Avoiding Dummy Variable Trap
# dataset = dataset[:, 1:]

# Feature Scaling
from sklearn.preprocessing import StandardScaler

#
scaler = MinMaxScaler(feature_range=(0, 2))
# scaler = StandardScaler()
# dataset = scaler.fit_transform(dataset)

# spliting the dataset into test data and training data
from sklearn.model_selection import train_test_split

training_set, test_set = train_test_split(dataset, test_size=0.1)

# Prepare Training Data
X_train, y_train = [], []

for i in range(6, training_set.shape[0] - 7):
    X_train.append(training_set[i - 6:i])
    y_train.append(training_set[i+1, 8])

X_train = np.array(X_train)
y_train = np.array(y_train)

# X_train = np.reshape(X_train.shape[0], X_train.shape[1], 1)
# y_train = np.reshape(y_train.shape[0], y_train.shape[1], 1)

# Build LSTM
regressor = Sequential()
regressor.add(LSTM(units=100, activation='relu', input_shape=(X_train.shape[1], 11), return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=170, activation='relu', return_sequences=True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units=190, activation='relu', return_sequences=True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units=250, activation='relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
regressor.fit(X_train, y_train, epochs=15, batch_size=10)

# prepare test set
training_set = pd.DataFrame(training_set)
test_set = pd.DataFrame(test_set)

# 6*10
past_60_seconds = training_set.tail(6)

test_set = past_60_seconds.append(test_set, ignore_index=True)

X_test, y_test = [], []

test_set = np.array(test_set)

for i in range(6, test_set.shape[0] - 6):
    X_test.append(test_set[i - 6:i])
    y_test.append(test_set[i, 8])

X_test = np.array(X_test)
y_test = np.array(y_test)

X_test_0 = X_train[0]
X_test_0 = np.array(X_test_0)
X_test_0 = X_test_0.reshape(1, 6, 11)
y_pred_0 = regressor.predict(X_test_0)

X_test_1 = X_train[1]
X_test_1 = np.array(X_test_1)
X_test_1 = X_test_1.reshape(1, 6, 11)
y_pred_1 = regressor.predict(X_test_1)

X_test_2 = X_train[2]
X_test_2 = np.array(X_test_2)
X_test_2 = X_test_2.reshape(1, 6, 11)
y_pred_2 = regressor.predict(X_test_2)


y_pred = regressor.predict(X_test)
