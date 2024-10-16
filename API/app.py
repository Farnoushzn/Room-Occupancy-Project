"""
# Room Occupancy Prediction API

This script sets up a Flask API to serve predictions for room occupancy based on sensor data (temperature, humidity, CO2, etc.).
The trained model (XGBoost Classifier) is loaded and used to provide predictions.

## Key Features:
- **API Endpoint:** `/predict` - Accepts POST requests with sensor data to provide occupancy predictions.
- **Input Format:** JSON containing the necessary sensor readings and one-hot encoded time features.
- **Output Format:** JSON response with the occupancy prediction and probability score.

## Prerequisites:
- Python 3.x
- Required packages as specified in the `requirements.txt` file:
  - Flask (for API)
  - joblib (for loading the trained model)
  - numpy, pandas (for data manipulation)

## Running the API:
1. Make sure you have all the dependencies installed.
2. Start the Flask server by running this script:
    ```sh
    python app.py
    ```
3. The API will be available locally at `http://127.0.0.1:5000` (Home Endpoint)

## Example Usage:
- **Home Endpoint:** You can check if the API is running by visiting `http://127.0.0.1:5000/`.
- **Prediction Endpoint:**
  - To get a room occupancy prediction, make a POST request to `http://127.0.0.1:5000/predict`(prediction endpoint) with appropriate sensor data in JSON format.


"""
# Import necessary libraries
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_path = "Models/xgb_occupancy_model.pkl"
model = joblib.load(model_path)

# Load columns used in model training (to ensure alignment)
feature_columns = [
    'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio',
    'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7',
    'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
    'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
    'hour_22', 'hour_23',
    'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 'day_of_week_3', 
    'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
    'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
    'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
]

# Define the home route
@app.route('/')
def home():
    return "Room Occupancy Prediction API is up and running!"

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.get_json()

    # Convert the incoming JSON data into a DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # Reindex to align input data with the expected feature columns, filling missing values with zero
    aligned_data = input_data.reindex(columns=feature_columns, fill_value=0)

    # Convert all columns to numeric type to ensure compatibility with XGBoost
    aligned_data = aligned_data.apply(pd.to_numeric, errors='coerce')

    # Make predictions using the trained model
    prediction = model.predict(aligned_data)
    prediction_prob = model.predict_proba(aligned_data)[:, 1]

    # Return the prediction as a JSON response
    response = {
        'prediction': int(prediction[0]),
        'prediction_probability': float(prediction_prob[0])
    }

    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
