import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define log file path
log_file_path = os.getenv('LOG_FILE_PATH', './Models/training_log.txt')
file_handler = logging.FileHandler(log_file_path, mode='w')  # Clear log file every time the script runs
logger.addHandler(file_handler)

try:
    # Load preprocessed datasets
    train_path = os.getenv('TRAIN_PATH', "app/Dataset/Processed_data/processed_train.csv")
    val_path = os.getenv('VAL_PATH', "app/Dataset/Processed_data/processed_val.csv")
    test_path = os.getenv('TEST_PATH', "app/Dataset/Processed_data/processed_test.csv")
    model_save_path = os.getenv('MODEL_SAVE_PATH', "app/Models/xgb_occupancy_model.pkl")  # Default to local path

    df_train_encoded = pd.read_csv(train_path)
    df_val_encoded = pd.read_csv(val_path)
    df_test_encoded = pd.read_csv(test_path)

    # Get the union of columns across all datasets
    all_columns = set(df_train_encoded.columns) | set(df_val_encoded.columns) | set(df_test_encoded.columns)

    # Ensure each dataset has the same columns, adding missing ones with zeros
    df_train_encoded = df_train_encoded.reindex(columns=all_columns, fill_value=0)
    df_val_encoded = df_val_encoded.reindex(columns=all_columns, fill_value=0)
    df_test_encoded = df_test_encoded.reindex(columns=all_columns, fill_value=0)

    # Display first few rows of the train dataset to verify
    # logger.info("Training Data with Unified Columns Across Datasets:")
    # logger.info(df_train_encoded.head().to_string())

    # Train XGBoost Classifier
    desired_feature_columns = [
        'Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio',
        'hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7',
        'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14',
        'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21',
        'hour_22', 'hour_23', 'day_of_week_0', 'day_of_week_1', 'day_of_week_2', 
        'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
        'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 
        'month_7', 'month_8', 'month_9', 'month_10', 'month_11', 'month_12'
    ]

    # Set feature columns
    feature_columns = [col for col in desired_feature_columns if col in df_train_encoded.columns]

    X_train = df_train_encoded[feature_columns]
    y_train = df_train_encoded['Occupancy']
    X_val = df_val_encoded[feature_columns]
    y_val = df_val_encoded['Occupancy']

    # Define and train the model
    logger.info("Starting model training...")
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    logger.info("Model training completed successfully.")

    # Evaluate the model
    y_val_pred_prob = xgb_model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(y_val, y_val_pred_prob)
    accuracy = accuracy_score(y_val, xgb_model.predict(X_val))

    # Log final scores
    logger.info(f"Validation ROC AUC Score: {roc_auc:.2f}")
    logger.info(f"Validation Accuracy Score: {accuracy:.2f}")

    # Hyperparameter Optimization : Random Search
    logger.info("Starting hyperparameter optimization with RandomizedSearchCV...")
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(eval_metric='auc', random_state=42),
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings sampled
        scoring='roc_auc',  # Evaluation metric
        cv=3,  # 3-fold cross-validation
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    random_search.fit(X_train, y_train)
    logger.info("Hyperparameter optimization completed.")
    logger.info(f"Best Parameters found by Random Search: {random_search.best_params_}")

    # Use the best estimator to predict on the validation set
    best_model = random_search.best_estimator_
    y_val_best_pred_prob = best_model.predict_proba(X_val)[:, 1]
    best_roc_auc = roc_auc_score(y_val, y_val_best_pred_prob)
    logger.info(f"Validation ROC AUC Score with Best Hyperparameters: {best_roc_auc:.2f}")

    # Save the trained model
    joblib.dump(best_model, model_save_path)
    logger.info(f"Best trained XGBoost model saved at: {model_save_path}")

    # Summary of Model Training Results
    logger.info("Summary of Model Training Results:")
    logger.info(f"Validation ROC AUC Score with Best Hyperparameters: {best_roc_auc:.2f}")
    accuracy = accuracy_score(y_val, best_model.predict(X_val))
    logger.info(f"Validation Accuracy Score: {accuracy:.2f}")
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_val, best_model.predict(X_val)))
    logger.info("Classification Report:\n%s", classification_report(y_val, best_model.predict(X_val)))

    # Plot the ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_val_best_pred_prob)
    roc_auc_value = auc(fpr, tpr)

    # Commenting the plotting code as requested
    # plt.figure(figsize=(10, 6))
    # plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_value:.2f})')
    # plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc='lower right')
    # plt.show()

except Exception as e:
    logger.error(f"Training failed: {e}")
