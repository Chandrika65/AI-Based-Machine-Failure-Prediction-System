
# ==========================================
# Save Best Machine Failure Prediction Model
# ==========================================

import os
import joblib


# Function to save the trained model
def save_model(best_model):

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    joblib.dump(
        best_model,
        "models/machine_failure_model.pkl"
    )

    print("\nBest model saved successfully!")
    print("Model location: models/machine_failure_model.pkl")


