import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math

# Load data
df = pd.read_csv("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/data/processed/data_clean.csv")

# K-NEAREST NEIGHBORS MODEL
print("=" * 60)
print("K-NEAREST NEIGHBORS MODEL")
print("=" * 60)

# Independent variables
features = ["Quantity", "Unit_Price", "Discount_Rate", "Revenue", "Cost"]
X = df[features]

# Target variable
y = df["Profit"]

print(f"\nFeatures: {features}")
print(f"Target: Profit")
print(f"Dataset size: {len(df)} records")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# Train the model
k_neighbors = 5
model = KNeighborsRegressor(n_neighbors=k_neighbors)
model.fit(X_train, y_train)

print(f"\nKNN model trained with k={k_neighbors}")

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_train = math.sqrt(mse_train)
rmse_test = math.sqrt(mse_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("\n" + "=" * 60)
print("MODEL RESULTS")
print("=" * 60)
print(f"\nTRAINING SET:")
print(f"  MAE (Mean Absolute Error): ${mae_train:.2f}")
print(f"  MSE (Mean Squared Error): ${mse_train:.2f}")
print(f"  RMSE (Root MSE): ${rmse_train:.2f}")
print(f"  R² Score: {r2_train:.4f}")

print(f"TEST SET:")
print(f"  MAE (Mean Absolute Error): ${mae_test:.2f}")
print(f"  MSE (Mean Squared Error): ${mse_test:.2f}")
print(f"  RMSE (Root MSE): ${rmse_test:.2f}")
print(f"  R² Score: {r2_test:.4f}")

# Test different k values
print("\n" + "=" * 60)
print("TESTING DIFFERENT K VALUES")
print("=" * 60)

k_values = [1, 3, 5, 7, 10, 15, 20]

for k in k_values:
    model_k = KNeighborsRegressor(n_neighbors=k)
    model_k.fit(X_train, y_train)
    y_pred_k = model_k.predict(X_test)
    r2_k = r2_score(y_test, y_pred_k)
    rmse_k = math.sqrt(mean_squared_error(y_test, y_pred_k))
    print(f"k={k:2d} -> R² = {r2_k:.4f} | RMSE = ${rmse_k:.2f}")
