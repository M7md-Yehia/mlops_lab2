import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('models/model.joblib')

# Load test data
test_df = pd.read_csv('data/test.csv')

# Separate features (X) and target (y)
X_test = test_df.drop('Outcome', axis=1)
y_test = test_df['Outcome']

# Make predictions
preds = model.predict(X_test)

# --- 1. Save Metrics (metrics.json) ---
acc = accuracy_score(y_test, preds)
with open('metrics.json', "w") as f:
    json.dump({'accuracy': acc}, f)
print(f"Metrics saved. Test Accuracy: {acc:.4f}")

# --- 2. Save Plot (confusion_matrix.png) ---
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'], 
            yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("Plot saved to confusion_matrix.png.")