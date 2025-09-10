import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and encoders
model = None
encoders = None
feature_names = None

def load_model_local():
    """Load model and encoders from local files"""
    try:
        global model, encoders, feature_names
        
        # Load the TensorFlow model
        model = tf.keras.models.load_model("resale_price_estimator_tf_converted")
        logger.info("TensorFlow model loaded successfully")
        
        # Load label encoders
        encoders = joblib.load("label_encoders.pkl")
        logger.info("Label encoders loaded successfully")
        
        # Load feature names
        feature_names = np.load("feature_names.npy", allow_pickle=True)
        logger.info(f"Feature names loaded: {len(feature_names)} features")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def preprocess_input(input_data):
    """Preprocess input data for prediction"""
    df = pd.DataFrame([input_data])

    # Binary mappings
    bool_cols = ["AuthenticityVerified", "TaxBenefitEligible", "AR_TryOnAvailable", "EcoFriendlyPackaging"]
    for col in bool_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # Ordinal encoding for ConditionGrade
    condition_ordinal_map = {
        'Fair': 1, 'Good': 2, 'Very Good': 3, 'Like New': 4, 'New': 5
    }
    df['ConditionGrade'] = df['ConditionGrade'].map(condition_ordinal_map).fillna(2).astype(int)

    # Ordinal encoding for DonationStatus
    donation_status_map = {"Available": 0, "Reserved": 1, "Sold": 2}
    df['DonationStatus'] = df['DonationStatus'].map(donation_status_map).fillna(0).astype(int)

    # Label encode categorical features
    categorical_cols = [
        "RetailerName", "Category", "SubCategory", "BrandName", "Size", "Color",
        "StorageLocation", "ShippingPartner", "MaterialType"
    ]
    global encoders
    for col in categorical_cols:
        if encoders is not None and col in encoders and df[col].iloc[0] in encoders[col].classes_:
            df[col] = encoders[col].transform([df[col].iloc[0]])
        else:
            df[col] = 0

    return df

def calculate_resale_price(row, target_margin):
    """Calculate resale price with profit margin"""
    min_price = row['OriginalPrice'] * (1 + target_margin / 100)
    base_price = max(row['PredictedPrice'], min_price)
    
    # Round up to nearest .99
    resale_price = np.floor(base_price) + 0.99
    if resale_price < base_price:
        resale_price = np.floor(base_price + 1) + 0.99
    
    return round(resale_price, 2)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "encoders_loaded": encoders is not None,
        "feature_count": len(feature_names) if feature_names is not None else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        global model
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        target_margin = data.get('TargetMargin', 5.0)
        processed = preprocess_input(data)
        features = processed.drop(columns=["TargetMargin"]) if "TargetMargin" in processed.columns else processed
        input_array = features.to_numpy().astype(np.float32)
        
        # Make prediction
        prediction = model.predict(input_array)[0][0]
        processed['PredictedPrice'] = prediction
        
        # Calculate resale price
        resale_price = calculate_resale_price(processed.iloc[0], target_margin)
        margin = (resale_price - data['OriginalPrice']) / data['OriginalPrice'] * 100
        
        response = {
            "predicted_base_price": float(prediction),
            "recommended_resale_price": resale_price,
            "achieved_profit_margin": round(margin, 2),
            "target_margin": target_margin,
            "original_price": data['OriginalPrice'],
            "model_type": "TensorFlow (converted from LightGBM)"
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        global model
        data = request.get_json()
        
        if not data or 'items' not in data:
            return jsonify({"error": "No items provided for batch prediction"}), 400
        
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        results = []
        for item in data['items']:
            try:
                # Process single item
                processed = preprocess_input(item)
                features = processed.drop(columns=["TargetMargin"]) if "TargetMargin" in processed.columns else processed
                input_array = features.to_numpy().astype(np.float32)
                
                prediction = model.predict(input_array)[0][0]
                processed['PredictedPrice'] = prediction
                
                target_margin = item.get('TargetMargin', 5.0)
                resale_price = calculate_resale_price(processed.iloc[0], target_margin)
                margin = (resale_price - item['OriginalPrice']) / item['OriginalPrice'] * 100
                
                results.append({
                    "input": item,
                    "predicted_base_price": float(prediction),
                    "recommended_resale_price": resale_price,
                    "achieved_profit_margin": round(margin, 2)
                })
                
            except Exception as e:
                results.append({
                    "input": item,
                    "error": str(e)
                })
        
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    global model, feature_names
    return jsonify({
        "model_type": "TensorFlow (converted from LightGBM)",
        "feature_count": len(feature_names) if feature_names is not None else 0,
        "feature_names": feature_names.tolist() if feature_names is not None else [],
        "model_summary": str(model.summary()) if model is not None else "Model not loaded"
    })

# Load model on startup (always, even with Gunicorn)
try:
    load_model_local()
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    exit(1)

if __name__ == '__main__':
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False) 