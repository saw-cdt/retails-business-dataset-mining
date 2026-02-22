#imposible no vuelvo a intentar usar vim con jupyter
import pandas as pd
import os

df = pd.read_csv("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/data/processed/data_clean.csv")

print("=== Descriptive Statistics ===")
print(df.describe())

print("\n=== Dataset Information ===")
print(df.info())

print("\n=== Columns (Entities and Attributes) ===")
for col in df.columns:
    print(f"- {col}: {df[col].dtype}")

print("\n=== Dispersion Measures ===")
numeric_cols = ['Quantity', 'Unit_Price', 'Discount_Rate', 'Revenue', 'Cost', 'Profit']
for col in numeric_cols:
    print(f"\n-- {col} --")
    std = df[col].std()
    var = df[col].var()
    col_min = df[col].min()
    col_max = df[col].max()
    rng = col_max - col_min
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    print(f"Standard Deviation: {std:.2f}")
    print(f"Variance: {var:.2f}")
    print(f"Range: {rng:.2f} (min: {col_min}, max: {col_max})")
    print(f"IQR (Q3-Q1): {iqr:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})")

print("\n=== Identified Entities ===")
print("1. Customer: Customer_ID, Customer_Segment")
print("2. Order/Transaction: Order_Date, Region, Payment_Method")
print("3. Product: Product_Category, Unit_Price, Quantity, Discount_Rate")
print("4. Finance: Revenue, Cost, Profit")

print("\n=== Relationships ===")
print("- A Customer places Orders/Transactions")
print("- A Transaction includes Products from a Category")
print("- Each row represents a transaction line with financial metrics")

print("\n=== Statistics Grouped by Product_Category ===")
grouped_category = df.groupby('Product_Category').agg({
    'Revenue': ['mean', 'sum', 'count'],
    'Profit': ['mean', 'sum'],
    'Quantity': 'sum'
})
print(grouped_category)

print("\n=== Statistics Grouped by Region ===")
grouped_region = df.groupby('Region').agg({
    'Revenue': ['mean', 'sum'],
    'Profit': ['mean', 'sum'],
    'Quantity': 'sum'
})
print(grouped_region)

print("\n=== Statistics Grouped by Customer_Segment ===")
grouped_segment = df.groupby('Customer_Segment').agg({
    'Revenue': ['mean', 'sum'],
    'Profit': ['mean', 'sum'],
    'Quantity': 'sum'
})
print(grouped_segment)