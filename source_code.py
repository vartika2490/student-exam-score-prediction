# main.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set Seaborn style for better visuals
sns.set(style="whitegrid")
sns.set_palette("summer")  # Nice green palette

# Define path to ZIP and target CSV
zip_path = r"C:/Users/Acer/Downloads/student_performance_dataset.zip"
extract_to = "extracted_data"
csv_filename = "student_performance_dataset.csv"

# Extract the ZIP file if not already extracted
if not os.path.exists(extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Load dataset
csv_path = os.path.join(extract_to, csv_filename)
df = pd.read_csv(csv_path)

# Drop unnecessary columns if any
df = df.drop(columns=["Student_ID", "Pass_Fail"], errors='ignore')

# Encode categorical variables (if applicable)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Internet_Access_at_Home'] = df['Internet_Access_at_Home'].map({'No': 0, 'Yes': 1})
df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'No': 0, 'Yes': 1})

# Select features and target
X = df[["Study_Hours_per_Week", "Attendance_Rate", "Past_Exam_Scores"]]
y = df["Final_Exam_Score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Enhanced Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, s=80, color="#8cc537", edgecolor="black")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label="Ideal Prediction")
plt.xlabel("Actual Exam Score", fontsize=12, color='black')
plt.ylabel("Predicted Exam Score", fontsize=12, color='pink')
plt.title("Actual vs Predicted Exam Scores", fontsize=14, fontweight='bold', color='darkgreen')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
