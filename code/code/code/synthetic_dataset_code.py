# ------------------------------------------------------------
# Machine Learning Model Comparison using Train–Test Split
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------
data = pd.read_csv(r"data/synthetic_mec_dataset.csv")
print("✅ Dataset loaded successfully!\n")
print("Shape:", data.shape)
print("Columns:", data.columns.tolist())

# ------------------------------------------------------------
# 2. Features (X) and target (y)
# ------------------------------------------------------------
X = data.select_dtypes(include=[np.number]).drop(columns=["Hydrogen Production Rate (HPR)"], errors='ignore')
y = data["Hydrogen Production Rate (HPR)"]

# ------------------------------------------------------------
# 3. Split the dataset (80% training, 20% testing)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n📊 Train–Test Split Done:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ------------------------------------------------------------
# 4. Define models
# ------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "kNN": KNeighborsRegressor(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Neural Network": MLPRegressor(max_iter=1000, random_state=42)
}

# ------------------------------------------------------------
# 5. Train and evaluate each model
# ------------------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append([name, round(r2, 3), round(rmse, 3), round(mae, 3)])
    print(f"{name} ✅ R²={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")

# ------------------------------------------------------------
# 6. Summary table
# ------------------------------------------------------------
results_df = pd.DataFrame(results, columns=["Model", "R²", "RMSE", "MAE"])
print("\n📊 Model Comparison using 80–20 Train–Test Split:\n")
print(results_df)
