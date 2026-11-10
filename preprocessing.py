
# ==================================================
# AI-BASED MACHINE FAILURE PREDICTION SYSTEM
# COMPLETE DATA PREPROCESSING
# ==================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np


# 2. LOAD DATASET

df = pd.read_csv("data/ai4i2020.csv")

print("\n========== ORIGINAL DATASET ==========")
print(df.head())

print("Original Shape:", df.shape)


# 3. REMOVE UNNECESSARY COLUMNS

df = df.drop(columns=[
    "UDI",
    "Product ID",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

print("\n========== AFTER REMOVING COLUMNS ==========")
print(df.head())

print("New Shape:", df.shape)


# 4. CHECK MISSING VALUES

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

print("Total Missing Values:",
      df.isnull().sum().sum())


# 5. HANDLE MISSING VALUES

numerical_columns = df.select_dtypes(
    include="number"
).columns

for column in numerical_columns:

    df[column] = df[column].fillna(
        df[column].median()
    )

print("\n========== AFTER MISSING VALUE HANDLING ==========")

print(df.isnull().sum())


# 6. REMOVE DUPLICATE RECORDS

print("\n========== DUPLICATE RECORDS ==========")

print("Before Removing:", df.duplicated().sum())

df = df.drop_duplicates()

print("After Removing:", df.duplicated().sum())

print("Shape After Removing Duplicates:", df.shape)


# 7. OUTLIER DETECTION USING IQR

print("\n========== OUTLIER DETECTION ==========")

outlier_columns = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

for column in outlier_columns:

    Q1 = df[column].quantile(0.25)

    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print("\nColumn:", column)

    print("Q1:", Q1)

    print("Q3:", Q3)

    print("IQR:", IQR)

    print("Lower Limit:", lower_limit)

    print("Upper Limit:", upper_limit)

    print("Number of Outliers:", len(outliers))


# IMPORTANT:
# We are detecting outliers but not deleting them.
# Extreme sensor readings may indicate machine failure.


# 8. FEATURE SELECTION

print("\n========== FEATURE SELECTION ==========")

X = df.drop(
    "Machine failure",
    axis=1
)

y = df["Machine failure"]

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

print("\nInput Shape:", X.shape)

print("Target Shape:", y.shape)


# 9. CATEGORICAL ENCODING

print("\n========== CATEGORICAL ENCODING ==========")

X = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

print(X.head())

print("\nColumns After Encoding:")
print(X.columns)


# 10. FINAL DATASET INFORMATION

print("\n========== FINAL DATASET ==========")

print("Input Shape:", X.shape)

print("Target Shape:", y.shape)

print("\nInput Information:")
print(X.info())

print("\nTarget Distribution:")
print(y.value_counts())


# 11. SAVE PROCESSED DATA

X.to_csv(
    "data/cleaned_features.csv",
    index=False
)

y.to_csv(
    "data/cleaned_target.csv",
    index=False
)

print("\n========== PREPROCESSING COMPLETED ==========")

print("Cleaned features saved successfully!")

print("Cleaned target saved successfully!")
