import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
file_path = r"D:\Python\LRM.csv"  # Replace with your actual CSV file
df = pd.read_csv(file_path)

# Drop rows with missing values in the output column (index 2)
df_cleaned = df.dropna(subset=[df.columns[2]])

# Extract input (X) and output (Y) columns
X = df_cleaned.iloc[:, [1]].values  # Column index 1 as input
Y = df_cleaned.iloc[:, 2].values    # Column index 2 as output

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Model evaluation
r2 = r2_score(Y_test, Y_pred)  # R² score (Accuracy of the model)

# Save the trained model
joblib.dump(model, "linear_model.pkl")

# Function to load trained model
def get_trained_model():
    return model

# Print results
# print(f"Model Coefficient: {model.coef_[0]}")
# print(f"Model Intercept: {model.intercept_}")
# print(f"Model Accuracy (R² Score): {r2}")

