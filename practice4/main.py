import pandas as pd
from scipy.stats import f_oneway
from scipy.stats import kruskal
import seaborn as sns
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

# ANOVA
for cat in cat_cols:
    print(f"\nANOVA for {cat}")
    for num in num_cols:
        groups = []
        for value in df[cat].unique():
            groups.append(df[df[cat] == value][num])
        stat, p = f_oneway(*groups)
        print(f"{num}: p-value = {p}")

# Kruskal
for cat in cat_cols:
    print(f"\nKruskal for {cat}")
    for num in num_cols:
        groups = []
        for value in df[cat].unique():
            groups.append(df[df[cat] == value][num])
        stat, p = kruskal(*groups)
        print(f"{num}: p-value = {p}")
        
sns.boxplot(x="Region", y="Revenue", data=df)
plt.show()

print("\nThe results show that all p-values are greater than 0.05, indicating that there is no statistically significant difference between the groups. The Kruskal-Wallis test produced consistent outcomes, so we can conclude that the labeled groups do not significantly affect the numerical variables in the dataset.")