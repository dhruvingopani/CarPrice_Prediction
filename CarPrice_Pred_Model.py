import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

# Data Collection and Precessing
car_dataset = pd.read_csv('car data.csv')
# print(car_dataset.head())

# Statistical Analysis
# print(car_dataset.shape)
# print(car_dataset.info())
# print(car_dataset.describe())
# print(car_dataset.isnull().sum())

# checking the distribution of categorical data
# print(car_dataset['Fuel_Type'].value_counts())
# print(car_dataset['Seller_Type'].value_counts())
# print(car_dataset['Transmission'].value_counts())

# Encoding the Categorical Data
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2},
                     'Seller_Type': {'Dealer': 0, 'Individual': 1},
                     'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)
# print(car_dataset.head(10))

# Splitting the data and Target
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
y = car_dataset['Selling_Price']
# print(X)
# print(y)

# Splitting training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training

# 1) Linear Regression
# model1 = LinearRegression()
# model1.fit(X_train, y_train)

# Model Evaluation
# training_data_prediction = model1.predict(X_train)
# error_score = r2_score(y_train, training_data_prediction)
# print('R squared error: ', error_score)

# visualize the actual prices and predicted prices
# plt.scatter(x=y_train, y=training_data_prediction)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual Prices vs Predicted Prices')
# plt.show()

# prediction on test data
# test_data_prediction = model1.predict(X_test)
# error_score = r2_score(y_test, test_data_prediction)
# print('R squared error: ', error_score)

# visualize the actual prices and predicted prices on test data
# plt.scatter(x=y_test, y=test_data_prediction)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual Prices vs Predicted Prices on test data')
# plt.show()

# 2) Lasso Regression
model2 = Lasso()
model2.fit(X_train, y_train)

# Model Evaluation
training_data_prediction = model2.predict(X_train)
# error_score = r2_score(y_train, training_data_prediction)
# print('R squared error: ', error_score)

# visualize the actual prices and predicted prices
plt.scatter(x=y_train, y=training_data_prediction)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()

# prediction on test data
# test_data_prediction = model1.predict(X_test)
# error_score = r2_score(y_test, test_data_prediction)
# print('R squared error: ', error_score)

# visualize the actual prices and predicted prices on test data
# plt.scatter(x=y_test, y=test_data_prediction)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual Prices vs Predicted Prices on test data')
# plt.show()