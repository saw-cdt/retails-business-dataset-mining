import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# CARGA Y AGRUPACIÓN MENSUAL
df = pd.read_csv('../data/processed/data_clean.csv')
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

df['Year_Month'] = df['Order_Date'].dt.to_period('M')
monthly = df.groupby('Year_Month').agg({
    'Revenue':       'sum',
    'Profit':        'sum',
    'Quantity':      'sum',
    'Discount_Rate': 'mean',
    'Unit_Price':    'mean'
}).reset_index()

monthly['month_num'] = range(1, len(monthly) + 1)

# ESTACIONALIDAD (seno/coseno)
monthly['sin_month'] = np.sin(2 * np.pi * monthly['month_num'] / 12)
monthly['cos_month'] = np.cos(2 * np.pi * monthly['month_num'] / 12)

feature_cols = ['month_num', 'Quantity', 'Discount_Rate', 'Unit_Price',
                'sin_month', 'cos_month']

X = monthly[feature_cols]
y = monthly['Revenue']

# TRAIN / TEST SPLIT
train_size = int(0.8 * len(monthly))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# PIPELINE: escalado + Ridge
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=1.0))
])
pipe.fit(X_train, y_train)

y_pred_train = pipe.predict(X_train)
y_pred_test  = pipe.predict(X_test)

cv_scores = cross_val_score(pipe, X, y, cv=3, scoring='r2')

print('=== Métricas (Modelo Corregido) ===')
print(f'Train R²:    {r2_score(y_train, y_pred_train):.4f}')
print(f'Test  R²:    {r2_score(y_test, y_pred_test):.4f}')
print(f'Test  MAE:   ${mean_absolute_error(y_test, y_pred_test):,.2f}')
print(f'Test  RMSE:  ${np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}')
print(f'CV R² (3-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# PREDICCIONES
future_months = [13, 14, 15]
future_df = pd.DataFrame({
    'month_num':     future_months,
    'Quantity':      [monthly['Quantity'].mean()] * 3,
    'Discount_Rate': [monthly['Discount_Rate'].mean()] * 3,
    'Unit_Price':    [monthly['Unit_Price'].mean()] * 3,
    'sin_month':     np.sin(2 * np.pi * np.array(future_months) / 12),
    'cos_month':     np.cos(2 * np.pi * np.array(future_months) / 12),
})

future_df = future_df[feature_cols]
future_preds = pipe.predict(future_df)

print('\n=== Predicciones Futuras (2025) ===')
for date, pred in zip(['Ene', 'Feb', 'Mar'], future_preds):
    print(f'{date} 2025: ${pred:,.2f}')


# PLOT
x_real   = np.arange(1, 13)
x_train  = np.arange(1, train_size + 1)
x_test   = np.arange(train_size + 1, 13)
x_future = np.arange(13, 16)

plt.figure(figsize=(13, 7))

plt.plot(x_real, monthly['Revenue'].values, 'o-', color='#2563EB',
         linewidth=2.5, markersize=10, label='Revenue Real', zorder=5)

plt.plot(x_train, y_pred_train, 's--', color='#F59E0B',
         linewidth=2, markersize=7, label='Predicción Train', alpha=0.85, zorder=4)

plt.plot(x_test, y_pred_test, 's--', color='#DC2626',
         linewidth=2, markersize=7, label='Predicción Test', alpha=0.85, zorder=4)

plt.plot(x_future, future_preds, '^--', color='#16A34A',
         linewidth=2.5, markersize=11, label='Forecast 2025', zorder=6)

plt.axvline(x=train_size + 0.5, color='#6B7280', linestyle=':',
            linewidth=2, alpha=0.8, label='Train / Test Split')

# Anotaciones revenue
for i, (x, y_val) in enumerate(zip(x_real, monthly['Revenue'].values)):
    if i % 2 == 0:
        plt.annotate(f'${y_val/1e6:.2f}M', (x, y_val),
                     textcoords="offset points", xytext=(0, -22),
                     ha='center', fontsize=9, fontweight='bold', color='#2563EB')

# Anotaciones forecast
for x, y_val in zip(x_future, future_preds):
    plt.annotate(f'${y_val/1e6:.2f}M', (x, y_val),
                 textcoords="offset points", xytext=(0, 12),
                 ha='center', fontsize=10, fontweight='bold', color='#16A34A')

month_labels = ['M1','M2','M3','M4','M5','M6','M7','M8',
                'M9','M10','M11','M12','M13\n(Ene)','M14\n(Feb)','M15\n(Mar)']
plt.xticks(np.arange(1, 16), month_labels, fontsize=10)
plt.yticks(fontsize=11)

plt.xlabel('Mes', fontsize=13, fontweight='bold')
plt.ylabel('Revenue ($)', fontsize=13, fontweight='bold')
plt.title('Forecasting de Revenue — Ridge Regression + Estacionalidad',
          fontsize=15, fontweight='bold', pad=15)

metrics_text = (f"Test R²:   {r2_score(y_test, y_pred_test):.3f}\n"
                f"Test MAE: ${mean_absolute_error(y_test, y_pred_test)/1e3:.1f}K\n"
                f"Test RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test))/1e3:.1f}K")
plt.text(0.02, 0.97, metrics_text, transform=plt.gca().transAxes,
         fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.legend(loc='upper right', fontsize=10, framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig('forecasting_plot.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
