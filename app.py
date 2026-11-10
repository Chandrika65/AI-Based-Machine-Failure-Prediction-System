
"""from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib


# 1. Create FastAPI application
app = FastAPI(
    title="AI Machine Failure Prediction System",
    description="Predict industrial machine failures using sensor data",
    version="1.0"
)


# 2. Load trained model
model = joblib.load("models/machine_failure_model.pkl")


# 3. Define input data structure
class MachineData(BaseModel):

    Type: str = Field(pattern="^[LMH]$")

    air_temperature: float = Field(
        gt=0,
        alias="Air temperature [K]"
    )

    process_temperature: float = Field(
        gt=0,
        alias="Process temperature [K]"
    )

    rotational_speed: float = Field(gt=0)

    torque: float = Field(ge=0)

    tool_wear: float = Field(ge=0)


# 4. Create home endpoint
@app.get("/")
def home():

    return {
        "message": "AI Machine Failure Prediction API is running"
    }


# 5. Create prediction endpoint
@app.post("/predict")
def predict_machine_failure(data: MachineData):

    # Convert input into DataFrame
    input_data = pd.DataFrame([{
        "Type": data.Type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    # Predict machine condition
    prediction = model.predict(input_data)[0]

    # Calculate failure probability
    probability = model.predict_proba(input_data)[0][1]

    # Prepare result
    if prediction == 1:
        status = "Machine Failure"
    else:
        status = "Machine Normal"

    return {
        "prediction": int(prediction),
        "status": status,
        "failure_probability": round(float(probability) * 100, 2),
        "message": "Prediction completed successfully"
    }
"""


import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Create FastAPI application
app = FastAPI(title="MachineGuard AI")

# Load trained ML model
model = joblib.load("models/machine_failure_model.pkl")


# Input data structure
class MachineData(BaseModel):
    machine_type: str
    air_temperature: float = Field(gt=0)
    process_temperature: float = Field(gt=0)
    rotational_speed: float = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: float = Field(ge=0)


# Webpage
@app.get("/", response_class=HTMLResponse)
def home():

    return """
<!DOCTYPE html>
<html>
<head>
    <title>MachineGuard AI</title>

    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #eef2f7;
            color: #172033;
        }

        header {
            background: #101d32;
            color: white;
            padding: 22px 8%;
        }

        header h1 {
            margin: 0;
            color: #54d6b0;
        }

        header p {
            margin-bottom: 0;
            color: #c5d1e0;
        }

        .container {
            max-width: 850px;
            margin: 35px auto;
            padding: 0 18px;
        }

        .card {
            background: white;
            padding: 28px;
            border-radius: 14px;
            box-shadow: 0 5px 20px #00000010;
        }

        h2 {
            margin-top: 0;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-bottom: 8px;
        }

        input, select {
            width: 100%;
            padding: 13px;
            border: 1px solid #ccd5e0;
            border-radius: 8px;
            font-size: 15px;
        }

        input:focus, select:focus {
            outline: 2px solid #54d6b0;
        }

        button {
            width: 100%;
            padding: 15px;
            margin-top: 25px;
            background: #153b36;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 17px;
            cursor: pointer;
        }

        button:hover {
            background: #20584e;
        }

        #result {
            display: none;
            margin-top: 25px;
            padding: 22px;
            border-radius: 10px;
            text-align: center;
        }

        .normal {
            background: #dcfce7;
            color: #166534;
        }

        .failure {
            background: #fee2e2;
            color: #991b1b;
        }

        .error {
            display: block !important;
            background: #fff1f2;
            color: #9f1239;
        }

        .result-title {
            font-size: 24px;
            font-weight: bold;
        }

        footer {
            text-align: center;
            padding: 25px;
            color: #64748b;
        }

        @media(max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }

            .card {
                padding: 20px;
            }
        }
    </style>
</head>

<body>

<header>
    <h1>⚙ MachineGuard AI</h1>
    <p>AI-Based Machine Failure Prediction System</p>
</header>

<div class="container">

    <div class="card">

        <h2>Machine Sensor Details</h2>
        <p>Enter the machine readings to predict possible failure.</p>

        <form id="predictionForm">

            <div class="form-grid">

                <div>
                    <label>Machine Type</label>
                    <select id="machine_type" required>
                        <option value="L">Low (L)</option>
                        <option value="M" selected>Medium (M)</option>
                        <option value="H">High (H)</option>
                    </select>
                </div>

                <div>
                    <label>Air Temperature (K)</label>
                    <input type="number" id="air_temperature"
                           value="298.1" step="any" min="0" required>
                </div>

                <div>
                    <label>Process Temperature (K)</label>
                    <input type="number" id="process_temperature"
                           value="308.6" step="any" min="0" required>
                </div>

                <div>
                    <label>Rotational Speed (rpm)</label>
                    <input type="number" id="rotational_speed"
                           value="1500" step="any" min="0" required>
                </div>

                <div>
                    <label>Torque (Nm)</label>
                    <input type="number" id="torque"
                           value="40" step="any" min="0" required>
                </div>

                <div>
                    <label>Tool Wear (min)</label>
                    <input type="number" id="tool_wear"
                           value="120" step="any" min="0" required>
                </div>

            </div>

            <button type="submit" id="predictButton">
                Predict Machine Failure
            </button>

        </form>

        <div id="result"></div>

    </div>

</div>

<footer>
    MachineGuard AI | Powered by Machine Learning
</footer>

<script>

const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function(event) {

    event.preventDefault();

    const button = document.getElementById("predictButton");
    const result = document.getElementById("result");

    button.disabled = true;
    button.innerText = "Predicting...";
    result.style.display = "block";
    result.className = "";
    result.innerText = "Analyzing machine data...";

    const data = {
        machine_type: document.getElementById("machine_type").value,
        air_temperature: Number(document.getElementById("air_temperature").value),
        process_temperature: Number(document.getElementById("process_temperature").value),
        rotational_speed: Number(document.getElementById("rotational_speed").value),
        torque: Number(document.getElementById("torque").value),
        tool_wear: Number(document.getElementById("tool_wear").value)
    };

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const output = await response.json();

        if (!response.ok) {
            throw new Error(output.detail || "Prediction failed");
        }

        result.className = output.prediction === 1
            ? "failure"
            : "normal";

        result.innerHTML = `
            <div class="result-title">${output.result}</div>
            <p>Failure probability: ${output.failure_probability}%</p>
            <p>${output.message}</p>
        `;

    } catch(error) {

        result.className = "error";
        result.innerText = error.message;

    } finally {

        button.disabled = false;
        button.innerText = "Predict Machine Failure";

    }

});

</script>

</body>
</html>
"""


# Prediction API
@app.post("/predict")
def predict(data: MachineData):

    if data.machine_type not in ["L", "M", "H"]:
        return {"error": "Invalid machine type"}

    input_data = pd.DataFrame([{
        "Type": data.machine_type,
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear
    }])

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]
    failure_index = list(model.classes_).index(1)
    failure_probability = float(probabilities[failure_index] * 100)

    if prediction == 1:
        result = "⚠ Machine Failure Predicted"
        message = "Potential failure detected. Inspect the machine."
    else:
        result = "✓ Machine Normal"
        message = "No failure predicted for these readings."

    return {
        "prediction": prediction,
        "result": result,
        "failure_probability": round(failure_probability, 2),
        "message": message
    }
