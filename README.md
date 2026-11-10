# ⚙️ MachineGuard AI
## AI-Based Machine Failure Prediction System

### Project Overview
MachineGuard AI is a machine learning-based predictive maintenance application that predicts potential machine failures using sensor readings.

It helps identify possible failures before they occur, supporting preventive maintenance and reducing unexpected machine downtime.

### Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Random Forest Classifier
- GridSearchCV
- FastAPI
- HTML, CSS and JavaScript

### Features
- Machine sensor data analysis
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Machine failure classification
- Random Forest model training
- Hyperparameter tuning using GridSearchCV
- Failure probability estimation
- Interactive web application
- Real-time prediction through FastAPI

### Input Features
1. Machine Type
2. Air Temperature
3. Process Temperature
4. Rotational Speed
5. Torque
6. Tool Wear

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 97.45% |
| Precision | 60% |
| Recall | 75% |
| F1 Score | 66.67% |

*Performance measured on the held-out test dataset.*

### How to Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn app:app --reload
```

Open the browser:

```text
http://127.0.0.1:8000
```

### Prediction Output
The application displays:
- Machine Normal
- Machine Failure Predicted
- Estimated failure probability

### Disclaimer
This project is an educational machine learning prototype. Predictions are estimates and should not be used as the sole basis for industrial safety decisions.

### Author
Chandrika

