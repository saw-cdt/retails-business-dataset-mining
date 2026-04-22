import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# Configure variables
K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RANDOM_STATE = 42

# Load data
df = pd.read_csv("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/data/processed/data_clean.csv")

print("=" * 60)
print("K-MEANS CLUSTERING MODEL")
print("=" * 60)

print(f"\nOriginal dataset: {len(df)} records")

# Define features for clustering (numeric variables)
features = ["Quantity", "Unit_Price", "Discount_Rate", "Revenue", "Cost", "Profit"]

# Aggregate data by customer: calculate mean of each feature per customer
customer_data = df.groupby("Customer_ID")[features].mean().reset_index()
print(f"Customer aggregation: {len(customer_data)} unique customers")

print(f"\nFeatures for clustering:")
for f in features:
    print(f"  - {f}")

# Prepare feature matrix
X = customer_data[features]

# Standardize features (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n" + "=" * 60)
print("ELBOW METHOD & SILHOUETTE ANALYSIS")
print("=" * 60)

inertias = []
silhouette_scores = []

# Test each K value
print(f"\n{'K':>3} | {'Inertia':>12} | {'Silhouette':>10}")
print("-" * 30)

for k in K_VALUES:
    # Fit K-Means model
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_scaled)
    
    # Store metrics
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(sil_score)
    
    print(f"{k:>3} | {kmeans.inertia_:>12.2f} | {sil_score:>10.4f}")

# Select optimal K (highest silhouette score)
best_k_idx = np.argmax(silhouette_scores)
BEST_K = K_VALUES[best_k_idx]

print("\n" + "=" * 60)
print(f"OPTIMAL K SELECTED: k={BEST_K}")
print("=" * 60)
print(f"\nBest Silhouette Score: {silhouette_scores[best_k_idx]:.4f}")

# Create Elbow Method and Silhouette Score plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Method plot
axes[0].plot(K_VALUES, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=BEST_K, color='r', linestyle='--', label=f'Optimal K={BEST_K}')
axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[0].set_ylabel('Inertia', fontsize=12)
axes[0].set_title('Elbow Method', fontsize=14)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Silhouette Score plot
axes[1].plot(K_VALUES, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=BEST_K, color='r', linestyle='--', label=f'Optimal K={BEST_K}')
axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Analysis', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/practice7/elbow_silhouette.png", dpi=150)
print("\nSaved: elbow_silhouette.png")

# Train final K-Means model with optimal K
final_model = KMeans(n_clusters=BEST_K, random_state=RANDOM_STATE, n_init=10)
customer_data['Cluster'] = final_model.fit_predict(X_scaled)

print("\n" + "=" * 60)
print("CLUSTER DISTRIBUTION")
print("=" * 60)

# Count customers per cluster
cluster_counts = customer_data['Cluster'].value_counts().sort_index()
print(f"\n{'Cluster':>8} | {'Customers':>12} | {'Percentage':>10}")
print("-" * 35)
for cluster, count in cluster_counts.items():
    pct = (count / len(customer_data)) * 100
    print(f"{cluster:>8} | {count:>12} | {pct:>9.1f}%")

print("\n" + "=" * 60)
print("CLUSTER PROFILES (Mean Values)")
print("=" * 60)

# Calculate mean values for each cluster
cluster_profiles = customer_data.groupby('Cluster')[features].mean()
print(f"\n{cluster_profiles.round(2).to_string()}")

print("\n" + "=" * 60)
print("CLUSTER VISUALIZATION")
print("=" * 60)

# Scatter plot: Revenue vs Profit by cluster
fig2, ax2 = plt.subplots(figsize=(10, 8))
scatter = ax2.scatter(
    customer_data['Revenue'], 
    customer_data['Profit'], 
    c=customer_data['Cluster'], 
    cmap='viridis', 
    alpha=0.6, 
    s=50
)

# Plot cluster centroids (transform back to original scale)
centers = scaler.inverse_transform(final_model.cluster_centers_)
ax2.scatter(centers[:, 3], centers[:, 5], c='red', marker='X', s=200, edgecolors='black', linewidths=2, label='Centroids')
ax2.set_xlabel('Revenue', fontsize=12)
ax2.set_ylabel('Profit', fontsize=12)
ax2.set_title(f'K-Means Clustering (K={BEST_K})', fontsize=14)
ax2.legend()
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Cluster')
plt.tight_layout()
plt.savefig("/Users/alexquintanilla/Documents/workspace/retails-business-dataset-mining/practice7/clusters_scatter.png", dpi=150)
print("\nSaved: clusters_scatter.png")

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
print(f"\n  Optimal K: {BEST_K}")
print(f"  Total Customers: {len(customer_data)}")
print(f"  Silhouette Score: {silhouette_scores[best_k_idx]:.4f}")
print(f"  Final Inertia: {final_model.inertia_:.2f}")
print("\n" + "=" * 60)
