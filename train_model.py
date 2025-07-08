import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

# Step 1: Generate training data
np.random.seed(42)
n = 30
X = pd.DataFrame({
    'BOD_raw': np.random.uniform(150, 300, n),
    'COD_raw': np.random.uniform(300, 600, n),
    'TSS_raw': np.random.uniform(100, 300, n),
    'Oil_and_Grease_raw': np.random.uniform(30, 60, n),
    'pH_raw': np.random.uniform(6.0, 8.5, n),
    'Ammonical_N_raw': np.random.uniform(10, 20, n),
    'Total_N_raw': np.random.uniform(20, 40, n)
})

y = pd.DataFrame({
    'BOD_treated': X['BOD_raw'] * 0.04,
    'COD_treated': X['COD_raw'] * 0.08,
    'TSS_treated': X['TSS_raw'] * 0.07,
    'Oil_and_Grease_treated': X['Oil_and_Grease_raw'] * 0.1,
    'pH_treated': np.random.uniform(6.5, 8.0, n),
    'Ammonical_N_treated': X['Ammonical_N_raw'] * 0.2,
    'Total_N_treated': X['Total_N_raw'] * 0.2
})

# Step 2: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Step 3: Save the trained model
with open("wastewater_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as 'wastewater_model.pkl'")
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Split the synthetic data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n🎯 Model Performance on Test Set:")
print(f"R² Score     : {r2:.4f}")
print(f"MSE          : {mse:.4f}")
print(f"MAE          : {mae:.4f}")
