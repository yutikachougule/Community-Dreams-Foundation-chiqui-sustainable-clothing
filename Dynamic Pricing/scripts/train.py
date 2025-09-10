import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load Feature-Engineered Dataset
df = pd.read_csv("../data/processed_dataset.csv")

# Drop irrelevant columns
drop_cols = ['DonationDate','ProductID', 'ProductName', 'ProductImageURL']
for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Define Target and Features
target = 'EstimatedResalePrice'
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in the dataset.")

# Encode categorical columns if not already encoded
categorical_cols = ['RetailerName', 'Category', 'SubCategory', 'BrandName',
                    'Size', 'Color', 'ConditionGrade', 'StorageLocation',
                    'ShippingPartner', 'MaterialType']

# Check if columns are numeric already to avoid double encoding
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

# Save encoders if any were created
if encoders:
    joblib.dump(encoders, '../models/label_encoders.pkl')
    print("Label encoders saved as label_encoders.pkl")
else:
    print("[i] No new encoding performed (columns already numeric or missing).")

# Split features and target
X = df.drop(columns=[target])
y = df[target]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Feature names: {list(X.columns)}\n")

# Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM Model
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
# rmse = mean_squared_error(y_val, y_pred, squared=False)

mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)

print(f"Validation RMSE: {rmse:.2f}")

# Save Model
joblib.dump(model, '../models/xgb_resale_model.pkl')
print("Model saved as xgb_resale_model.pkl")
