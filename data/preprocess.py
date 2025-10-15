import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

df = pd.read_csv('D:\mlops\mlops_lab2\data\diabetes.csv') 

# --- DATA CLEANING AND FEATURE ENGINEERING ---

# 1. Rename columns for clarity (if not already named)
# This assumes the columns are named as standard: Pregnancies, Glucose, BloodPressure, etc.
# If your downloaded file has different headers, adjust this section.
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# 2. Identify features where zero values are biologically impossible
# These are commonly BloodPressure, BMI, Glucose, SkinThickness, and Insulin.
cols_to_replace = ['BloodPressure', 'BMI', 'Glucose', 'SkinThickness', 'Insulin']

# 3. Replace 0s in these columns with NaN (Not a Number)
df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

# 4. Impute Missing Values: Fill NaNs with the mean of their respective columns
# This is a simple imputation technique.
for col in cols_to_replace:
    df[col].fillna(df[col].mean(), inplace=True)


# 5. Separate features (X) and target (y) (Target column is 'Outcome')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 6. Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Recombine X and y to save as full CSVs for the pipeline
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

# Save the output files required by the assignment
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("Preprocessing complete. Saved data/train.csv and data/test.csv.")