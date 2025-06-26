import numpy as np
from LRM import get_trained_model  # Import trained model from model.py

# Load the trained model
model = get_trained_model()

def predict_value(input_value):
    """Predict the output for a given input value using the trained model."""
    input_array = np.array([[input_value]])  # Convert to 2D array
    prediction = model.predict(input_array)
    return prediction[0]

# Example usage when imported
if __name__ == "__main__":
    new_input = float(input("Enter a new input value: "))  # User input
    result = predict_value(new_input)
    # print(f"Predicted Output: {result}")
