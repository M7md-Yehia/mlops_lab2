import pandas as pd
from sklearn.model_selection import train_test_split
import os
import numpy as np

df = pd.read_csv('D:\mlops\mlops_lab2\data\diabetes.csv') 


df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']


cols_to_replace = ['BloodPressure', 'BMI', 'Glucose', 'SkinThickness', 'Insulin']

df[cols_to_replace] = df[cols_to_replace].replace(0, np.nan)

for col in cols_to_replace:
    df[col].fillna(df[col].mean(), inplace=True)


X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

os.makedirs('data', exist_ok=True)

train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("Preprocessing complete. Saved data/train.csv and data/test.csv.")