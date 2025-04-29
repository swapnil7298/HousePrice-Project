from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler

class Linear_Regression():
    def __init__(self, learning_rate=0.01, no_of_iterations=1000):
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

app = Flask(__name__)

house_price_dataset = sklearn.datasets.fetch_california_housing()
house_df = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_df['price'] = house_price_dataset.target

X = house_df.drop(['price'], axis=1)
Y = house_df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Linear_Regression(learning_rate=0.01, no_of_iterations=3000)
model.fit(X_scaled, Y)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_values = [
            float(request.form['MedInc']),
            float(request.form['HouseAge']),
            float(request.form['AveRooms']),
            float(request.form['AveOccup']),
            float(request.form['Latitude']),
            float(request.form['Longitude'])
        ]
        input_values += [0.0, 0.0]  

        input_values_scaled = scaler.transform([input_values])

        prediction = model.predict(input_values_scaled)[0]

        return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
