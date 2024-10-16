# %% [markdown]
# # Model Training and Evaluation
# 
# This notebook is dedicated to training, optimizing, and evaluating the model for predicting room occupancy. Steps included:
# 
# 1. **Initial Model Training**:
#    - Train an **XGBoost Classifier** using default hyperparameters to establish a baseline.
#    - Evaluate the model on the validation set using **ROC AUC score**.
# 
# 2. **Hyperparameter Optimization**:
#    - Use **Random Search** (`RandomizedSearchCV`) to find the optimal set of hyperparameters for the XGBoost model.
#    - The model is tuned to maximize **ROC AUC** using a parameter grid and cross-validation.
# 
# 3. **Evaluation of the Optimized Model**:
#    - Evaluate the **best model** obtained from Random Search using **accuracy**, **confusion matrix**, **classification report**, and **ROC Curve** to assess the performance.
#    - **Save the trained model** using **joblib** for future use or deployment.
# 
# 4. **Model Saving**:
#    - Save the optimized XGBoost model in a serialized format (`xgb_occupancy_model.pkl`) for use in deployment.
# 

# %% [markdown]
# ## Import Required Libraries
# 

# %%
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Model training and hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Model evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

# Saving the model
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns



# %% [markdown]
# ## Load Preprocessed Datasets

# %%
# Assuming preprocessed datasets are saved in CSV format or available from the previous step
train_path = "./Dataset/Processed_data/processed_train.csv"
val_path = "./Dataset/Processed_data/processed_val.csv"
test_path = "./Dataset/Processed_data/processed_test.csv"

# Load preprocessed data
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
print("Training Data with Unified Columns Across Datasets:")
print(df_train_encoded.head())



# %% [markdown]
# ## Train XGBoost Classifier
# 

# %%

# Define the desired feature columns
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

# Set the feature columns based on actual available columns in training data
feature_columns = [col for col in desired_feature_columns if col in df_train_encoded.columns]

# Use the defined feature columns for training and validation
X_train = df_train_encoded[feature_columns]
y_train = df_train_encoded['Occupancy']

X_val = df_val_encoded[feature_columns]
y_val = df_val_encoded['Occupancy']

# Define the XGBoost Classifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

# Train the model on the training dataset
xgb_model.fit(X_train, y_train)





# %% [markdown]
# ## Predict on validation set
# 

# %%
# Predict on the validation set
y_val_pred = xgb_model.predict(X_val)
y_val_pred_prob = xgb_model.predict_proba(X_val)[:, 1]

# Evaluate the model using ROC AUC
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_val, y_val_pred_prob)
print(f"Validation ROC AUC Score: {roc_auc:.2f}")

# %% [markdown]
# ## Hyperparameter Optimization : Random Search

# %%
from sklearn.model_selection import RandomizedSearchCV

# Define parameter grid for Random Search
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Set up RandomizedSearchCV with XGBoost Classifier
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

# Fit the Random Search on the training data
random_search.fit(X_train, y_train)

# Print the best parameters found by Random Search
print("Best Parameters found by Random Search:")
print(random_search.best_params_)

# Use the best estimator to predict on the validation set
best_model = random_search.best_estimator_
y_val_best_pred_prob = best_model.predict_proba(X_val)[:, 1]

# Evaluate using ROC AUC
best_roc_auc = roc_auc_score(y_val, y_val_best_pred_prob)
print(f"Validation ROC AUC Score with Best Hyperparameters: {best_roc_auc:.2f}")


# %% [markdown]
# ## Save the Best-Trained Model

# %%
import joblib

# Define the path to save the best model
model_save_path = "./Models/xgb_occupancy_model.pkl"

# Save the trained model
joblib.dump(best_model, model_save_path)

print("Best trained XGBoost model saved successfully at:", model_save_path)



# %% [markdown]
# ## Summary of Model Training Results
# - Best Hyperparameters found by Random Search.
# - Validation ROC AUC Score for the best model.
# - Confusion Matrix and Classification Report for detailed evaluation.
# 

# %%
import matplotlib.pyplot as plt

# Print the best hyperparameters found by Random Search
print("Best Hyperparameters found by Random Search:")
print(random_search.best_params_)

# Calculate Validation ROC AUC Score with Best Hyperparameters
y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
best_roc_auc = roc_auc_score(y_val, y_val_pred_prob)
print("\nValidation ROC AUC Score with Best Hyperparameters: {:.2f}".format(best_roc_auc))

# Calculate accuracy score
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy Score: {accuracy:.2f}")

# Generate and print the Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_val, best_model.predict(X_val)))

# Generate and print the Classification Report
print("\nClassification Report:")
print(classification_report(y_val, best_model.predict(X_val)))

# Plot the ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_val_pred_prob)
roc_auc_value = auc(fpr, tpr)

# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc_value:.2f})')
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
#plt.show()


