from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

house_price_datsaset = sklearn.datasets.fetch_california_housing()

house_df = pd.DataFrame(house_price_datsaset.data, columns=house_price_datsaset.feature_names)

house_df['price'] = house_price_datsaset.target

house_df.isnull().sum()

correlation = house_df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')

X = house_df.drop(['price'], axis=1)
Y = house_df['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

iterations = 1000 
learning_rate = 0.0002

class CustomLinearRegression():
    def __init__(self, learning_rate, no_of_iterations):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)
        dw = -(2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = -2 * np.sum(self.Y - Y_prediction) / self.m
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b

model = CustomLinearRegression(learning_rate=learning_rate, no_of_iterations=iterations)
model.fit(X_train, Y_train)

class Lasso_Regression():
    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range(self.no_of_iterations):
            self.update_weights()

    def update_weights(self):
        Y_prediction = self.predict(self.X)
        dw = np.zeros(self.n)

        for i in range(self.n):
            if self.w[i] > 0:
                dw[i] = (-(2 * (self.X[:, i]).dot(self.Y - Y_prediction)) + self.lambda_parameter) / self.m
            else:
                dw[i] = (-(2 * (self.X[:, i]).dot(self.Y - Y_prediction)) - self.lambda_parameter) / self.m

        db = -2 * np.sum(self.Y - Y_prediction) / self.m
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db

    def predict(self, X):
        return X.dot(self.w) + self.b

model1 = Lasso_Regression(learning_rate=0.02, no_of_iterations=10000, lambda_parameter=100)
model1.fit(X_train, Y_train)

training_data_prediction = model1.predict(X_train)
test_data_prediction = model1.predict(X_test)

score_1_train = metrics.r2_score(Y_train, training_data_prediction)
score_2_train = metrics.mean_absolute_error(Y_train, training_data_prediction)
print("R squared error (training): ", score_1_train)
print("Mean absolute error (training): ", score_2_train)

score_1_test = metrics.r2_score(Y_test, test_data_prediction)
score_2_test = metrics.mean_absolute_error(Y_test, test_data_prediction)
print("R squared error (test): ", score_1_test)
print("Mean absolute error (test): ", score_2_test)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices (Training)")
plt.show()

plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs Predicted prices (Test)")
plt.show()
