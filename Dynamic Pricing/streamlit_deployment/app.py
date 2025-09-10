import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib

# Load Model and LabelEncoders
@st.cache_resource
def load_model():
    model = joblib.load("../models/lgbm_resale_model.pkl")  # Trained model
    encoders = joblib.load("../models/label_encoders.pkl")  # Categorical encoders
    return model, encoders

model, encoders = load_model()

# UI
st.title("Resale Price Estimator with Profit Margin")
st.sidebar.header("Product Details")

input_data = {
    "OriginalPrice": st.sidebar.number_input("Original Price", min_value=1.0),
    "RetailerName": st.sidebar.text_input("Retailer Name"),
    "Category": st.sidebar.text_input("Category"),
    "SubCategory": st.sidebar.text_input("Sub-Category"),
    "BrandName": st.sidebar.text_input("Brand Name"),
    "Size": st.sidebar.text_input("Size"),
    "Color": st.sidebar.text_input("Color"),
    "ConditionGrade": st.sidebar.selectbox("Condition Grade", ['Fair', 'Good', 'Very Good', 'Like New', 'New']),
    "DonationMonth": st.sidebar.selectbox("Donation Month", list(range(1, 13))),
    "DonationYear": st.sidebar.number_input("Donation Year", min_value=2000, max_value=2100, value=2024),
    "StorageLocation": st.sidebar.text_input("Storage Location"),
    "AuthenticityVerified": st.sidebar.selectbox("Authenticity Verified?", ["Yes", "No"]),
    "DonationStatus": st.sidebar.selectbox("Donation Status", ["Available", "Reserved", "Sold"]),
    "SustainabilityScore": st.sidebar.slider("Sustainability Score", 0.0, 1.0, 0.5),
    "TaxBenefitEligible": st.sidebar.selectbox("Tax Benefit Eligible?", ["Yes", "No"]),
    "WinningBidShippingCost": st.sidebar.number_input("Shipping Cost", min_value=0.0),
    "ShippingPartner": st.sidebar.text_input("Shipping Partner"),
    "AR_TryOnAvailable": st.sidebar.selectbox("AR Try-On?", ["Yes", "No"]),
    "RewardPoints": st.sidebar.number_input("Reward Points", min_value=0),
    "MaterialType": st.sidebar.text_input("Material Type"),
    "EcoFriendlyPackaging": st.sidebar.selectbox("Eco Packaging?", ["Yes", "No"]),
    "TargetMargin": st.sidebar.slider("Target Profit Margin (%)", 0.0, 100.0, 5.0),
}

# Preprocessing
def preprocess_input(input_data, encoders):
    df = pd.DataFrame([input_data])

    # Binary mappings
    bool_cols = ["AuthenticityVerified", "TaxBenefitEligible", "AR_TryOnAvailable", "EcoFriendlyPackaging"]
    for col in bool_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Ordinal encoding for ConditionGrade
    condition_ordinal_map = {
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Like New': 4,
        'New': 5
    }
    df['ConditionGrade'] = df['ConditionGrade'].map(condition_ordinal_map).fillna(2).astype(int)

    # Ordinal encoding for DonationStatus
    donation_status_map = {
        "Available": 0,
        "Reserved": 1,
        "Sold": 2
    }
    df['DonationStatus'] = df['DonationStatus'].map(donation_status_map).fillna(0).astype(int)

    # Label encode categoricals
    categorical_cols = [
        "RetailerName", "Category", "SubCategory", "BrandName", "Size", "Color",
        "StorageLocation", "ShippingPartner", "MaterialType"
    ]
    for col in categorical_cols:
        if col in encoders and df[col].iloc[0] in encoders[col].classes_:
            df[col] = encoders[col].transform([df[col].iloc[0]])
        else:
            df[col] = 0  # Unknown category fallback

    return df

# Prediction & Margin Logic
def calculate_resale_price(row, target_margin):
    min_price = row['OriginalPrice'] * (1 + target_margin / 100)
    base_price = max(row['PredictedPrice'], min_price)
    
    # Always round up to the next .99
    resale_price = np.floor(base_price) + 0.99
    if resale_price < base_price:
        resale_price = np.floor(base_price + 1) + 0.99
    
    return round(resale_price, 2)

    #return round(max(row['PredictedPrice'], min_price), 2)


# Action
if st.button("Estimate Resale Price"):
    processed = preprocess_input(input_data, encoders)
    features = processed.drop(columns=["TargetMargin"])  # Drop extra column

    prediction = model.predict(features)[0]
    processed['PredictedPrice'] = prediction

    resale_price = calculate_resale_price(processed.iloc[0], input_data["TargetMargin"])
    margin = (resale_price - input_data['OriginalPrice']) / input_data['OriginalPrice'] * 100

    st.success(f"Recommended Resale Price: **${resale_price}**")
    st.write(f"Predicted Base Price (no margin): ${prediction:.2f}")
    st.write(f"Achieved Profit Margin: {margin:.2f}%")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import joblib

# # Load Model and LabelEncoders
# @st.cache_resource
# def load_model():
#     model = tf.keras.models.load_model("resale_price_estimator_tf")  # TensorFlow SavedModel folder
#     encoders = joblib.load("label_encoders.pkl")
#     return model, encoders

# model, encoders = load_model()

# # UI
# st.title("Resale Price Estimator with Profit Margin")
# st.sidebar.header("Product Details")

# input_data = {
#     "OriginalPrice": st.sidebar.number_input("Original Price", min_value=1.0),
#     "RetailerName": st.sidebar.text_input("Retailer Name"),
#     "Category": st.sidebar.text_input("Category"),
#     "SubCategory": st.sidebar.text_input("Sub-Category"),
#     "BrandName": st.sidebar.text_input("Brand Name"),
#     "Size": st.sidebar.text_input("Size"),
#     "Color": st.sidebar.text_input("Color"),
#     "ConditionGrade": st.sidebar.selectbox("Condition Grade", ['Fair', 'Good', 'Very Good', 'Like New', 'New']),
#     "DonationMonth": st.sidebar.selectbox("Donation Month", list(range(1, 13))),
#     "DonationYear": st.sidebar.number_input("Donation Year", min_value=2000, max_value=2100, value=2024),
#     "StorageLocation": st.sidebar.text_input("Storage Location"),
#     "AuthenticityVerified": st.sidebar.selectbox("Authenticity Verified?", ["Yes", "No"]),
#     "DonationStatus": st.sidebar.selectbox("Donation Status", ["Available", "Reserved", "Sold"]),
#     "SustainabilityScore": st.sidebar.slider("Sustainability Score", 0.0, 1.0, 0.5),
#     "TaxBenefitEligible": st.sidebar.selectbox("Tax Benefit Eligible?", ["Yes", "No"]),
#     "WinningBidShippingCost": st.sidebar.number_input("Shipping Cost", min_value=0.0),
#     "ShippingPartner": st.sidebar.text_input("Shipping Partner"),
#     "AR_TryOnAvailable": st.sidebar.selectbox("AR Try-On?", ["Yes", "No"]),
#     "RewardPoints": st.sidebar.number_input("Reward Points", min_value=0),
#     "MaterialType": st.sidebar.text_input("Material Type"),
#     "EcoFriendlyPackaging": st.sidebar.selectbox("Eco Packaging?", ["Yes", "No"]),
#     "TargetMargin": st.sidebar.slider("Target Profit Margin (%)", 0.0, 100.0, 5.0),
# }

# # Preprocessing
# def preprocess_input(input_data, encoders):
#     df = pd.DataFrame([input_data])

#     # Binary mappings
#     bool_cols = ["AuthenticityVerified", "TaxBenefitEligible", "AR_TryOnAvailable", "EcoFriendlyPackaging"]
#     for col in bool_cols:
#         df[col] = df[col].map({"Yes": 1, "No": 0})

#     # Ordinal encoding for ConditionGrade
#     condition_ordinal_map = {
#         'Fair': 1,
#         'Good': 2,
#         'Very Good': 3,
#         'Like New': 4,
#         'New': 5
#     }
#     df['ConditionGrade'] = df['ConditionGrade'].map(condition_ordinal_map).fillna(2).astype(int)

#     # Ordinal encoding for DonationStatus
#     donation_status_map = {
#         "Available": 0,
#         "Reserved": 1,
#         "Sold": 2
#     }
#     df['DonationStatus'] = df['DonationStatus'].map(donation_status_map).fillna(0).astype(int)

#     # Label encode categorical features
#     categorical_cols = [
#         "RetailerName", "Category", "SubCategory", "BrandName", "Size", "Color",
#         "StorageLocation", "ShippingPartner", "MaterialType"
#     ]
#     for col in categorical_cols:
#         if col in encoders and df[col].iloc[0] in encoders[col].classes_:
#             df[col] = encoders[col].transform([df[col].iloc[0]])
#         else:
#             df[col] = 0  # Fallback for unknown values

#     return df

# # Prediction & Margin Logic
# def calculate_resale_price(row, target_margin):
#     min_price = row['OriginalPrice'] * (1 + target_margin / 100)
#     base_price = max(row['PredictedPrice'], min_price)
    
#     # Round up to nearest .99
#     resale_price = np.floor(base_price) + 0.99
#     if resale_price < base_price:
#         resale_price = np.floor(base_price + 1) + 0.99

#     return round(resale_price, 2)

# # Streamlit Action
# if st.button("Estimate Resale Price"):
#     processed = preprocess_input(input_data, encoders)
#     features = processed.drop(columns=["TargetMargin"])
#     input_array = features.to_numpy().astype(np.float32)

#     # Predict using TensorFlow model
#     prediction = model.predict(input_array)[0][0]
#     processed['PredictedPrice'] = prediction

#     resale_price = calculate_resale_price(processed.iloc[0], input_data["TargetMargin"])
#     margin = (resale_price - input_data['OriginalPrice']) / input_data['OriginalPrice'] * 100

#     st.success(f"Recommended Resale Price: **${resale_price}**")
#     st.write(f"Predicted Base Price (no margin): ${prediction:.2f}")
#     st.write(f"Achieved Profit Margin: {margin:.2f}%")
