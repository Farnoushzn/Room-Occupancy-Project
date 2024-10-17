import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Load environment variables for paths (similar to ModelTraining.py)
test_path = os.getenv('TEST_PATH', "app/Dataset/Processed_data/processed_test.csv")
model_path = os.getenv('MODEL_SAVE_PATH', "app/Models/xgb_occupancy_model.pkl")

try:
    # Load the saved model
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Load the test dataset
    df_test_encoded = pd.read_csv(test_path)

    # Define the feature columns (keeping in mind the ones used during training)
    feature_columns = [
        'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio',
        'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7',
        'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
        'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
        'hour_22', 'hour_23', 'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 
        'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
        'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 
        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
    ]

    # Filter the available columns in the test dataset to match the model's feature columns
    feature_columns = [col for col in feature_columns if col in df_test_encoded.columns]

    # Split features and target
    X_test = df_test_encoded[feature_columns]
    y_test = df_test_encoded['Occupancy']

    # Make predictions on the test dataset
    y_test_pred = model.predict(X_test)
    y_test_pred_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_prob)

    print(f"Test Accuracy Score: {accuracy:.2f}")
    print(f"Test ROC AUC Score: {roc_auc:.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

except Exception as e:
    print(f"Error during model testing: {e}")
