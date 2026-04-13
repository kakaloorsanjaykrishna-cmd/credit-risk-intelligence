import joblib
import os
import numpy as np

# Path to the saved model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "credit_model.pkl"))

def predict(input_data):
    """
    Takes a dictionary of customer data and returns a risk assessment.
    """
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return "Error: Model file not found. Run train.py first."

        # Load the pre-trained model
        model = joblib.load(MODEL_PATH)
        
        # Convert dictionary to numpy array in the correct order
        # Ensure the order matches the training features
        features = np.array([[
            input_data["Age"],
            input_data["Income"],
            input_data["CreditScore"],
            input_data["Utilization"],
            input_data["PaymentHistory"],
            input_data["ExistingLoans"],
            input_data["DefaultHistory"],
            input_data["EmploymentYears"]
        ]])

        prediction = model.predict(features)[0]
        
        # Return professional strings
        return "Low Risk (Approved)" if prediction == 0 else "High Risk (Rejected)"
    
    except Exception as e:
        return f"Prediction Error: {str(e)}"