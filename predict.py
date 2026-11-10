
import joblib
import pandas as pd


# 1. Load saved model
model = joblib.load("models/machine_failure_model.pkl")


# 2. Take new machine sensor readings
new_machine = pd.DataFrame([{
    "Type": "M",
    "Air temperature [K]": 300.5,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 45.0,
    "Tool wear [min]": 200
}])


# 3. Predict machine failure
prediction = model.predict(new_machine)


# 4. Get failure probability
probability = model.predict_proba(new_machine)[0][1]


# 5. Display result
print("\n========== MACHINE FAILURE PREDICTION ==========")

if prediction[0] == 1:
    print("Prediction: Machine Failure")
else:
    print("Prediction: Machine Normal")

print("Failure Probability:", round(probability * 100, 2), "%")
