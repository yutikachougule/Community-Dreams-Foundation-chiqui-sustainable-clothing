import joblib 

model = joblib.load("../models/lgbm_resale_model.pkl")
print("Model feature count:", model.n_features_)