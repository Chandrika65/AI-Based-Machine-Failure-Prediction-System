
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# 2. Remove unnecessary and leakage columns
df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

# 3. Separate input and target
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# 4. Identify columns
numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

categorical_features = ["Type"]

# 5. Create numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Create categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 9. Fit only on training data
X_train_scaled = preprocessor.fit_transform(X_train)

# 10. Transform testing data
X_test_scaled = preprocessor.transform(X_test)

# 11. Display results
print("Training shape:", X_train_scaled.shape)
print("Testing shape:", X_test_scaled.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nTrain-test split and scaling completed!")
