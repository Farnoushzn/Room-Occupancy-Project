# %% [markdown]
# # Data Exploration and Preprocessing
# 
# This notebook focuses on the preprocessing steps required for the Room Occupancy Prediction Project. It includes:
# 
# 1. **Data Loading**:
#    - Load the training, validation, and test datasets using Pandas.
# 
# 2. **Data Preprocessing and Feature Engineering**:
#    - Convert the `date` column to datetime format.
#    - Extract useful date-time features (`hour`, `day_of_week`, `month`).
#    - Apply **one-hot encoding** to the extracted categorical features (`hour`, `day_of_week`, `month`).
#    - Handle **missing values**, **outliers**, and **normalize** numerical features.
# 
# 3. **Splitting Data into Features and Labels**:
#    - Split the dataset into **features** (`X`) and **labels** (`y`) for training, validation, and testing.
# 

# %% [markdown]
# ## Load the Datasets Using Pandas

# %% [markdown]
# 

# %%
import pandas  as  pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# %%
# Define paths for each dataset
data_training_path = "../Dataset/occupancy_data/datatraining.txt"
data_validation_path = "../Dataset/occupancy_data/datatest.txt"
data_test_path = "../Dataset/occupancy_data/datatest2.txt"

# Load datasets
df_train = pd.read_csv(data_training_path)
df_val = pd.read_csv(data_validation_path)
df_test = pd.read_csv(data_test_path)

# Display the first few rows of each dataset to understand them
print("Training Data:")
display(df_train.head())
print("\nValidation Data:")
display(df_val.head())
print("\nTest Data:")
display(df_test.head())



# %% [markdown]
# ## Investigate the Attribute Information

# %%
# Path to the attribute information file
attribute_info_path = "../Dataset/occupancy_data/attribute_information.txt"

# Open and read the file
with open(attribute_info_path, 'r') as file:
    attribute_info = file.read()

# Print the attribute information to understand the features
print(attribute_info)


# %% [markdown]
# ## Data Exploration For Feature Engineering

# %%
## data exprloring
df_train.info()
df_train.describe()

# %%
# Check for missing values in the dataset
missing_values = df_train.isnull().sum()
print("Missing values per column:\n", missing_values)


# %% [markdown]
# ## Data Visualization

# %%
# Plot histograms for numerical features
df_train.hist(figsize=(12, 10), bins=30)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Extract categorical features from the datetime feature

# %%
# Convert 'date' column to datetime format
df_train['date'] = pd.to_datetime(df_train['date'])
df_val['date'] = pd.to_datetime(df_val['date'])
df_test['date'] = pd.to_datetime(df_test['date'])

# Extract useful date-time features
for df in [df_train, df_val, df_test]:
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

# Display the first few rows to verify the new features
df_train.head()


# %%

# Correlation heatmap plot to visualize relationships between features
# Drop only the original 'date' column but keep the extracted date-time features
df_train_numeric = df_train.drop(columns=['date'])

# Plot a correlation heatmap including extracted date-time features
plt.figure(figsize=(12, 10))
sns.heatmap(df_train_numeric.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap Including Date-Time Features")
plt.show()



# %%
# Generate box plots for each numerical feature to understand distributions between occupied and unoccupied states
for column in df_train_numeric.columns:
    if column != 'Occupancy':
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Occupancy', y=column, data=df_train, hue='Occupancy', palette='coolwarm', legend=False)
        plt.title(f"Box Plot of {column} by Occupancy")
        plt.show()


# %% [markdown]
# ## One-Hot Encode the Extracted Features

# %%

# Features to be one-hot encoded
categorical_features = ['hour', 'day_of_week', 'month']

# Apply one-hot encoding to categorical features for training, validation, and test datasets
df_train_encoded = pd.get_dummies(df_train, columns=categorical_features)
df_val_encoded = pd.get_dummies(df_val, columns=categorical_features)
df_test_encoded = pd.get_dummies(df_test, columns=categorical_features)

# Define all possible columns for hours, days, and months
all_hour_columns = ['hour_' + str(i) for i in range(24)]
all_day_columns = ['day_of_week_' + str(i) for i in range(7)]
all_month_columns = ['month_' + str(i) for i in range(1, 13)]
all_possible_columns = all_hour_columns + all_day_columns + all_month_columns

# Add missing columns across datasets if not already present
for df in [df_train_encoded, df_val_encoded, df_test_encoded]:
    for col in all_possible_columns:
        if col not in df.columns:
            df[col] = 0

# Ensure that validation and test datasets have the same columns as the training dataset
df_val_encoded = df_val_encoded.reindex(columns=df_train_encoded.columns, fill_value=0)
df_test_encoded = df_test_encoded.reindex(columns=df_train_encoded.columns, fill_value=0)

# Display the first few rows to verify encoding
print("Training Data after One-Hot Encoding and Adding Missing Columns:")
display(df_train_encoded.head())



# %% [markdown]
# ## Define Preprocessing Functions

# %% [markdown]
# ### Handle Missing Values:

# %%
def handle_missing_values(df):
    """
    Handles missing values in the given DataFrame.
    For numerical features like Temperature, Humidity, Light, etc., we will impute
    by grouping by 'hour' where applicable.
    
    Args:
        df (DataFrame): The DataFrame to handle missing values for.
        
    Returns:
        DataFrame: DataFrame with missing values handled.
    """
    for column in df.columns:
        if df[column].isnull().sum() > 0:  # Only handle columns with missing values
            if df[column].dtype in ['float64', 'int64']:
                # Impute using mean, grouped by 'hour' (context-sensitive imputation)
                df[column] = df.groupby('hour')[column].transform(lambda x: x.fillna(x.mean()))
            else:
                # Fill categorical values (if any) with mode
                df[column].fillna(df[column].mode()[0], inplace=True)

    return df


# %% [markdown]
# ### Handling Outliers

# %%
def handle_outliers(df):
    """
    Detects and handles outliers in the dataset using the IQR method,
    but preserves extreme values if they represent realistic data points.
    
    Args:
        df (DataFrame): The DataFrame to detect and handle outliers for.
        
    Returns:
        DataFrame: DataFrame with outliers selectively handled.
    """
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_features:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Selectively cap outliers based on feature context
        if column in ['Temperature', 'Humidity']:
            # Cap outliers for Temperature and Humidity, which should be stable
            df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
            df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
        # Light and CO2 might be more variable and important for prediction, keep those as is.

    return df


# %% [markdown]
# ### Normalizing Features

# %%

def normalize_features(df):
    """
    Normalizes numerical features using MinMax scaling.
    
    Args:
        df (DataFrame): The DataFrame to normalize features for.
        
    Returns:
        DataFrame: DataFrame with normalized numerical features.
    """
    numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

    scaler = MinMaxScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df


# %%
#Combine all preprocessing steps into one function.
def preprocess_data(df):
    df = handle_missing_values(df)
    df = handle_outliers(df)
    df = normalize_features(df)
    return df


# %% [markdown]
# ## Apply Preprocessing to All Datasets

# %%

# Apply the combined preprocessing function to the one-hot encoded datasets
df_train_final = preprocess_data(df_train_encoded)
df_val_final = preprocess_data(df_val_encoded)
df_test_final = preprocess_data(df_test_encoded)

# Drop the original `date` column as it is no longer needed
df_train_final = df_train_final.drop(columns=['date'])
df_val_final = df_val_final.drop(columns=['date'])
df_test_final = df_test_final.drop(columns=['date'])

# Display first few rows of the processed training data to verify
print("Final Preprocessed Training Data:")
print(df_train_final.head())

print("Validation Data after Preprocessing:")
print(df_val_final.head())

print("Test Data after Preprocessing:")
print(df_test_final.head())


# %% [markdown]
# ## Save Preprocessed Datasets

# %%
# Define paths for saving the preprocessed data
train_save_path = "../Dataset/Processed_data/processed_train.csv"
val_save_path = "../Dataset/Processed_data/processed_val.csv"
test_save_path = "../Dataset/Processed_data/processed_test.csv"

# Save the final preprocessed datasets
df_train_final.to_csv(train_save_path, index=False)
df_val_final.to_csv(val_save_path, index=False)
df_test_final.to_csv(test_save_path, index=False)

print("Preprocessed datasets saved successfully.")




