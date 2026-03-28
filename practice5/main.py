import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# load dataset
df = pd.read_csv(
    "/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/data/processed/data_clean.csv"
)

# features used to predict revenue
features = ["Quantity", "Unit_Price", "Discount_Rate"]
target   = "Revenue"

# split dataset into inputs (X) and target (y)
X = df[features]
y = df[target]

# select numeric columns for correlation analysis
num_cols = ["Quantity", "Unit_Price", "Discount_Rate", "Revenue", "Cost", "Profit"]
corr = df[num_cols].corr()  # correlation matrix

# plot correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,  # show correlation values
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    vmin=-1, vmax=1,
)
plt.title("Correlation Heatmap — Numeric Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("practice5/correlation_heatmap.png", dpi=150)
plt.show()

# create scatter plots for each feature vs revenue
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Feature Relationships with Revenue", fontsize=14, fontweight="bold")

colors = ["#4C72B0", "#DD8452", "#55A868"]

# loop through features and plot relationship with revenue
for ax, feat, color in zip(axes, features, colors):
    ax.scatter(df[feat], df[target], alpha=0.25, s=10, color=color)  # scatter points
    
    # fit a simple linear trend line
    m, b = np.polyfit(df[feat], df[target], 1)
    x_line = np.linspace(df[feat].min(), df[feat].max(), 200)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5, label="Trend")
    
    ax.set_xlabel(feat)
    ax.set_ylabel(target)
    ax.set_title(f"{feat} vs {target}")
    ax.legend()

plt.tight_layout()
plt.savefig("practice5/scatter_plots.png", dpi=150)
plt.show()

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create linear regression model
model = LinearRegression()
model.fit(X_train, y_train)  # train model

# print model parameters
print("=== Linear Regression Model ===")
print(f"Intercept : {model.intercept_:.4f}")

# print coefficients for each feature
for feat, coef in zip(features, model.coef_):
    print(f"  {feat:15s}: {coef:.4f}")

# make predictions on test set
y_pred = model.predict(X_test)

# evaluate model performance
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTest R²   : {r2:.4f}")
print(f"Test RMSE : {rmse:.4f}")

# plot actual vs predicted values
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.3, s=12, color="#4C72B0", label="Predictions")

# line representing perfect predictions
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")

plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title(f"Actual vs Predicted Revenue\n$R^2 = {r2:.4f}$", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig("practice5/actual_vs_predicted.png", dpi=150)
plt.show()

# compute residuals (errors)
residuals = y_test - y_pred

# plot residuals vs predictions
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.3, s=12, color="#DD8452")

# horizontal line at zero error
plt.axhline(0, color="red", linewidth=1.5, linestyle="--")

plt.xlabel("Predicted Revenue")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Revenue", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("practice5/residuals_plot.png", dpi=150)
plt.show()

# final summary of the model
print("\n=== Summary ===")
print(
    f"A multiple linear regression model was trained to predict Revenue "
    f"using {', '.join(features)}.\n"
    f"The model achieved an R² of {r2:.4f} on the test set, meaning it explains "
    f"{r2*100:.1f}% of the variance in Revenue.\n"
    f"The RMSE of {rmse:.2f} indicates the average prediction error in revenue units.\n"
    f"Quantity has the strongest positive relationship with Revenue, "
    f"consistent with the high correlation observed in the heatmap."
)