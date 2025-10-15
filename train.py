import pandas as pd
from sklearn.svm import SVC
import joblib
import os

# Load training data
train_df = pd.read_csv('data/train.csv')

# Separate features (X) and target (y)
X_train = train_df.drop('Outcome', axis=1)
y_train = train_df['Outcome']

model = SVC(random_state=42)
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')

print("Model training complete. Model saved to models/model.joblib.")