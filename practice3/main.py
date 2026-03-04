import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/data/processed/data_clean.csv")

# Define numeric columns
num_cols = [
    "Quantity",
    "Unit_Price",
    "Discount_Rate",
    "Revenue",
    "Cost", # Cost is redundant, cause we have revenue and profit but i'll include it anyways
    "Profit"
]

# Define categoric columns
cat_cols = [
    "Region",
    "Product_Category",
    "Customer_Segment",
    "Payment_Method"
]

# Histogram
for col in num_cols:
    plt.figure(figsize=(7, 4)) 
    plt.hist(df[col], bins=20, facecolor="#7E1E9C", edgecolor="black", alpha=0.7)
    plt.title(f"{col} Histogram")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.5)
    plt.show() 
    
# Boxplots
for col in num_cols:
    plt.figure(figsize=(7,4))
    plt.boxplot(df[col], patch_artist=True,
                boxprops=dict(facecolor="#DDA0DD", color="black"),
                medianprops=dict(color="red", linewidth=3))
    plt.title(f"{col} Boxplot")
    plt.ylabel(col)
    plt.show()

# Bar chart
for col in cat_cols:
    # Here we count hoy many times each category appears but sortde from highest to lowest
    counts = df[col].value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8,4))
    plt.bar(counts.index, counts.values, edgecolor = "white", facecolor="#DDA0DD")
    plt.title(f"{col} Bar Chart")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Pie chart
for col in cat_cols:
    counts = df[col].value_counts()
    plt.figure(figsize=(6,6))
    plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
    plt.title(f"{col} Pie Chart")
    plt.tight_layout()
    plt.show()

# Scatter  
# To do the scatter graph in loop as the practice says, im gonna make an array of pairs to do the loop
scatter_pairs = [
    ("Revenue", "Profit"),
    ("Discount_Rate", "Profit"),
    ("Quantity", "Revenue")
]

for x, y in scatter_pairs:
    plt.figure(figsize=(7,5))
    plt.scatter(df[x], df[y], alpha=0.3, s=10)
    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()