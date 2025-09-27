# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Step 1: Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
print("✅ Dataset loaded successfully!")

# Step 2: Explore the data
print(df.head())
print("\nDataset shape:", df.shape)
print("\nTarget classes:", iris.target_names)

# Step 3: Clean the data (Iris dataset is already clean)
print("\nMissing values:\n", df.isnull().sum())

# Step 4: Select features (X) and target (y)
X = df.drop("target", axis=1)
y = df["target"]

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining set:", X_train.shape)
print("Test set:", X_test.shape)

# Step 6: Feature scaling (improves model performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the model (Logistic Regression)
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)
print("\n✅ Model training complete!")

# Step 8: Make predictions
y_pred = model.predict(X_test_scaled)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\n📊 Model Performance:")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 10: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Iris Flower Classification')
plt.show()
