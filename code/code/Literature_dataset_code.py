import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Load dataset
df = pd.read_csv("data/literature dataset.csv", encoding="latin1")

# Drop extra unnamed columns if they exist
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
print("Columns in dataset:", df.columns)

# Define features and target
X = df.drop("Hydrogen Production Rate", axis=1)
y = df["Hydrogen Production Rate"]

# Handle missing values (replace NaN with mean for numeric, most frequent for categorical)
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(X[col].mean())

for col in X.select_dtypes(exclude=[np.number]).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Preprocessor: scale numeric + one-hot encode categorical
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Models dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "kNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
}

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# Train and evaluate each model
for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append([name, r2, rmse, mae])

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "R2 Score", "RMSE", "MAE"])
print("\nModel Performance:\n")
print(results_df)
