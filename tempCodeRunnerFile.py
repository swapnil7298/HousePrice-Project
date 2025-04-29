from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset and train the model (same steps you already have)
house_price_dataset = sklearn.datasets.fetch_california_housing()
house_df = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)
house_df['price'] = house_price_dataset.target

X = house_df.drop(['price'], axis=1)
Y = house_df['price']

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model using the LinearRegression model from sklearn
model = LinearRegression()
model.fit(X, Y)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input values from the form
        input_values = [float(request.form['MedInc']), float(request.form['HouseAge']),
                        float(request.form['AveRooms']), float(request.form['AveOccup']),
                        float(request.form['Latitude']), float(request.form['Longitude']),
                        float(request.form['MedHouseVal'])]
        
        # Standardize the input values using the scaler
        input_values_scaled = scaler.transform([input_values])

        # Predict the house price
        prediction = model.predict(input_values_scaled)[0]

        return render_template('index.html', prediction=prediction)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
