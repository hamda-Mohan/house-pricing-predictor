from utils import prepare_features_from_raw
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# --------------------------------
# Load the cleaned housing dataset
# --------------------------------
csv_path = 'dataset/Clean_house_dataset.csv'
df = pd.read_csv(csv_path)
# --------------------------------
# Define  features (X) and target (y)
# Target = Price, Features = all other columns except Price and LogPrice
# --------------------------------
x= df.drop(columns=["Price","LogPrice"])
y = df["Price"]
# --------------------------------
# Split data into training and testing sets
# 80% training, 20% testing for fair evaluation
# random_state ensures reproducibility
# --------------------------------
X_train, X_test ,y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) 
# --------------------------------
# Initialize and train Linear Regression model
# Linear Regression is a simple model assuming a linear relationship between features and target
# --------------------------------
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test) # Predict prices on the test set
# --------------------------------
# Initialize and train Random Forest model
# Random Forest is an ensemble method combining multiple decision trees for better accuracy
# --------------------------------
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test) # Predict prices on the test set
# --------------------------------
# Function to display performance metrics
# --------------------------------
def print_metrics(name, y, y_predict): # Metrics help us compare model performance:
    r2 = r2_score(y, y_predict) # R² indicates how well the model explains variance
    mae = mean_absolute_error(y, y_predict) # MAE shows average absolute error
    mse = mean_squared_error(y, y_predict) # MSE shows squared error penalizing large mistakes
    rmse = np.sqrt(mse) # RMSE is the square root of MSE, in same units as target
    print(f"Prediction of {name}")
    print(f"R2 : {r2:.3f}")
    print(f"MAE : {mae:,.0f}")
    print(f"MSE : {mse:,.0f}")
    print(f"RMSE : {rmse:,.0f}")
    print("-"*40)
# --------------------------------
# Evaluate both models using the test set
# --------------------------------
print_metrics("Linear Regression",y_test, lr_pred)
print_metrics("Random Forest",y_test, rf_pred)
# --------------------------------
# Check predictions on three individual test samples
# This is a sanity check to see how predictions compare to actual prices
# --------------------------------
sample_indices = [2,4, 6]

for i in sample_indices:
    x_one = X_test.iloc[[i]]  # select single row as DataFrame
    y_true = y_test.iloc[i]   # actual price
    p_lr = float(lr.predict(x_one)[0])  # Linear Regression prediction
    p_rf = float(rf.predict(x_one)[0])  # Random Forest prediction
    
    print(f"Single-row sanity check (Sample {i}):")
    print(f"  Actual Price: ${y_true:,.0f}")
    print(f"  LR Prediction: ${p_lr:,.0f}")
    print(f"  RF Prediction: ${p_rf:,.0f}")
    print("-"*40)

# --------------------------------
# SAVE MODELS (NEW)
# --------------------------------
joblib.dump(lr, "models/lr_model.joblib")
joblib.dump(rf, "models/rf_model.joblib")
print("\nSaved models → models/lr_model.joblib and models/rf_model.joblib")

# --------------------------------
# Optional: local custom input test using the shared helper
# --------------------------------
custom = {
    "Size_sqft": 2400, "Bedrooms": 3, "Bathrooms": 2,
    "YearBuilt": 2010, "Location": "City"
}
x_new_df = prepare_features_from_raw(custom)
print("\n=== Custom Input Prediction ===")
print("Linear Regression:", float(lr.predict(x_new_df)[0]))
print("Random Forest    :", float(rf.predict(x_new_df)[0]))