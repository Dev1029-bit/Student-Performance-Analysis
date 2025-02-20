# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import kagglehub

# Download Dataset from Kaggle
path = kagglehub.dataset_download("spscientist/students-performance-in-exams")
print("Path to dataset files:", path)

# Load Dataset
data = pd.read_csv(f"{path}/StudentsPerformance.csv")

# Data Preprocessing
# Select Relevant Features
selected_features = ['math score', 'reading score', 'writing score', 'lunch', 'test preparation course']

# Encode Categorical Features
data['lunch'] = data['lunch'].map({'standard': 0, 'free/reduced': 1})
data['test preparation course'] = data['test preparation course'].map({'none': 0, 'completed': 1})

X = data[selected_features]

# Target Column (Pass/Fail)
data['Pass_Fail'] = np.where((data[['math score', 'reading score', 'writing score']].mean(axis=1) >= 50), 1, 0) 
y = data['Pass_Fail']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the Model as .pkl
joblib.dump(model, "model/student_performance_model.pkl")
print("Model saved as student_performance_model.pkl")
