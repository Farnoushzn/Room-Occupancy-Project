"""
# Room Occupancy Prediction API Test Script

This script is used to test the Flask API created in `app.py` by sending a sample set of sensor data and receiving a room occupancy prediction.

## Key Features:
- **Endpoint Tested:** `/predict` - Sends a POST request with sensor readings in JSON format.
- **Input Format:** Sensor data including temperature, humidity, CO2, and one-hot encoded features (hour, day_of_week, and month).

## Prerequisites:
- Python 3.x
- Required packages as specified in `requirements.txt`:
  - `requests` (for making HTTP requests)
  - `json` (for handling JSON data)

## Running the Test:
1. Ensure the Flask API (`app.py`) is running locally at `http://127.0.0.1:5000`.
2. Run this script to send a prediction request:
    ```sh
    python prediction_req.py
    ```
3. The script will output:
   - **Status Code:** Response status from the API.
   - **Response Content:** JSON with the predicted room occupancy (0 or 1) and the prediction probability.

## Example Output:
- If successful, the output will look like:
Status Code: 200
Response Content: {
  "prediction": 0,
  "prediction_probability": 0.43084803223609924
}
"""

import requests
import json

# URL for the locally running Flask API
url = "http://127.0.0.1:5000/predict"

# Sample input data as a Python dictionary
# Ensure the data aligns with the features expected by the model (e.g., one-hot encoded features)
input_data = {
    "Temperature": 23.18,
    "Humidity": 27.272,
    "Light": 426,
    "CO2": 721.25,
    "HumidityRatio": 0.00479298817650529,
    "hour_17": 1,        # Example for one-hot encoded hour feature
    "day_of_week_3": 1,  # Example for one-hot encoded day of the week
    "month_2": 1         # Example for one-hot encoded month feature
}

# Make the POST request with JSON payload
response = requests.post(url, json=input_data)

# Print the status code and response content for debugging
print(f"Status Code: {response.status_code}")
print(f"Response Content: {response.text}")

# Try to parse the JSON response if available
try:
    response_json = response.json()
    print("Prediction Response:", response_json)
except json.JSONDecodeError:
    print("Error decoding the JSON response.")
