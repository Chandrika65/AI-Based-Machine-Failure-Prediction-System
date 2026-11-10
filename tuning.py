
import pandas as pd
import os
import joblib

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
df = pd.read_csv("data/ai4i2020.csv")

print("Dataset Shape:", df.shape)

# 2. Remove unnecessary columns
df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Separate input and output
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# 5. Define numerical and categorical columns
numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

categorical_features = ["Type"]

# 6. Numerical preprocessing
numerical_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

# 7. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 8. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 9. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 10. Create complete pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    ))
])

# 11. Define hyperparameter combinations
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

# 12. Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# 13. Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=1,
    verbose=1,
    error_score="raise"
)

# 14. Train and tune the model
print("\nStarting Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

# 15. Display best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation F1 Score:")
print(grid_search.best_score_)

# 16. Get best model
best_model = grid_search.best_estimator_

# 17. Predict test data
y_pred = best_model.predict(X_test)

# 18. Evaluate final model
print("\n========== FINAL MODEL EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))

print(
    "Precision:",
    precision_score(y_test, y_pred, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, y_pred, zero_division=0)
)

print(
    "F1 Score:",
    f1_score(y_test, y_pred, zero_division=0)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 19. Create models folder
os.makedirs("models", exist_ok=True)

# 20. Save best trained model
joblib.dump(
    best_model,
    "models/machine_failure_model.pkl"
)

print("\nBest model saved successfully!")

