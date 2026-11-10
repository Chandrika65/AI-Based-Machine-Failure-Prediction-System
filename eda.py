
# ============================================
# AI-BASED MACHINE FAILURE PREDICTION SYSTEM
# COMPLETE EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 2. LOAD DATASET

df = pd.read_csv("data/ai4i2020.csv")

print("\n========== DATASET LOADED ==========")
print("Dataset loaded successfully")


# 3. DISPLAY FIRST FIVE ROWS

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())


# 4. DISPLAY LAST FIVE ROWS

print("\n========== LAST FIVE ROWS ==========")
print(df.tail())


# 5. DATASET SHAPE

print("\n========== DATASET SHAPE ==========")
print("Rows and Columns:", df.shape)


# 6. COLUMN NAMES

print("\n========== COLUMN NAMES ==========")
print(df.columns)


# 7. DATASET INFORMATION

print("\n========== DATASET INFORMATION ==========")
df.info()


# 8. STATISTICAL ANALYSIS

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())


# 9. INDIVIDUAL STATISTICS

print("\n========== INDIVIDUAL STATISTICS ==========")

print("Average Air Temperature:",
      df["Air temperature [K]"].mean())

print("Maximum Air Temperature:",
      df["Air temperature [K]"].max())

print("Minimum Air Temperature:",
      df["Air temperature [K]"].min())

print("Average Process Temperature:",
      df["Process temperature [K]"].mean())

print("Average Rotational Speed:",
      df["Rotational speed [rpm]"].mean())

print("Average Torque:",
      df["Torque [Nm]"].mean())

print("Maximum Tool Wear:",
      df["Tool wear [min]"].max())


# 10. MISSING VALUES

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("Total Missing Values:",
      df.isnull().sum().sum())


# 11. DUPLICATE VALUES

print("\n========== DUPLICATE VALUES ==========")
print("Total Duplicates:", df.duplicated().sum())


# 12. DATA TYPES

print("\n========== DATA TYPES ==========")
print(df.dtypes)


# 13. UNIQUE VALUES

print("\n========== UNIQUE VALUES ==========")

print("Machine Types:", df["Type"].unique())

print("Number of Machine Types:",
      df["Type"].nunique())


# 14. MACHINE FAILURE COUNT

print("\n========== MACHINE FAILURE COUNT ==========")
print(df["Machine failure"].value_counts())


# 15. MACHINE FAILURE PERCENTAGE

print("\n========== MACHINE FAILURE PERCENTAGE ==========")
print(df["Machine failure"].value_counts(normalize=True) * 100)


# ============================================
# DATA VISUALIZATION
# ============================================


# 16. HISTOGRAM - TEMPERATURE DISTRIBUTION

plt.figure(figsize=(8, 5))

plt.hist(df["Air temperature [K]"], bins=30)

plt.title("Air Temperature Distribution")
plt.xlabel("Air Temperature")
plt.ylabel("Frequency")

plt.show()


# 17. COUNT PLOT - MACHINE FAILURE

plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Machine failure")

plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure")
plt.ylabel("Count")

plt.show()


# 18. BOXPLOT - OUTLIER DETECTION

plt.figure(figsize=(8, 5))

sns.boxplot(x=df["Torque [Nm]"])

plt.title("Torque Outlier Detection")
plt.xlabel("Torque")

plt.show()


# 19. SCATTERPLOT - SPEED VS TORQUE

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Rotational speed [rpm]",
    y="Torque [Nm]",
    hue="Machine failure"
)

plt.title("Rotational Speed vs Torque")

plt.show()


# 20. HEATMAP - CORRELATION

numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(12, 8))

correlation = numeric_df.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.show()


# 21. TOOL WEAR DISTRIBUTION

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="Tool wear [min]",
    bins=30,
    kde=True
)

plt.title("Tool Wear Distribution")

plt.show()


# 22. TEMPERATURE VS MACHINE FAILURE

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Machine failure",
    y="Air temperature [K]"
)

plt.title("Temperature vs Machine Failure")

plt.show()


# ============================================
# EDA COMPLETED
# ============================================

print("\n====================================")
print("EXPLORATORY DATA ANALYSIS COMPLETED")
print("====================================")

