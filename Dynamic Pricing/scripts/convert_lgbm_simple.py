import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import os

def convert_lgbm_to_tensorflow():
    """Convert LightGBM model to TensorFlow using a simpler approach"""
    print("🚀 Converting LightGBM to TensorFlow...")
    
    # Load the LightGBM model
    lgb_model = joblib.load("../models/lgbm_resale_model.pkl")
    encoders = joblib.load("../models/label_encoders.pkl")
    
    # Create a TensorFlow model that directly mimics LightGBM
    def create_lgbm_equivalent_model():
        """Create a TensorFlow model that approximates LightGBM behavior"""
        
        # Get feature names from LightGBM model
        feature_names = lgb_model.feature_name_
        input_shape = len(feature_names)
        
        print(f"📊 Creating model with {input_shape} input features")
        
        # Create a simple but effective neural network
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(input_shape,), name='input_features'),
            
            # Hidden layers - designed to capture non-linear relationships
            tf.keras.layers.Dense(512, activation='relu', name='dense_1'),
            tf.keras.layers.BatchNormalization(name='batch_norm_1'),
            tf.keras.layers.Dropout(0.3, name='dropout_1'),
            
            tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
            tf.keras.layers.BatchNormalization(name='batch_norm_2'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            
            tf.keras.layers.Dense(128, activation='relu', name='dense_3'),
            tf.keras.layers.BatchNormalization(name='batch_norm_3'),
            tf.keras.layers.Dropout(0.1, name='dropout_3'),
            
            # Output layer
            tf.keras.layers.Dense(1, activation='linear', name='output')
        ])
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model, feature_names
    
    # Create the TensorFlow model
    tf_model, feature_names = create_lgbm_equivalent_model()
    
    # Generate synthetic training data based on LightGBM predictions
    print("🔄 Generating training data...")
    
    # Create synthetic data that covers the feature space
    n_samples = 10000
    synthetic_data = np.random.randn(n_samples, len(feature_names))
    
    # Get LightGBM predictions for synthetic data
    lgb_predictions = lgb_model.predict(synthetic_data)
    
    # Train TensorFlow model to mimic LightGBM
    print("🎯 Training TensorFlow model...")
    history = tf_model.fit(
        synthetic_data, lgb_predictions,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        verbose=1
    )
    
    # Test with real data if available
    if os.path.exists("../data/processed_dataset.csv"):
        print("🧪 Testing with real data...")
        test_data = pd.read_csv("../data/processed_dataset.csv")
        
        # Preprocess test data (simplified version)
        test_features = preprocess_for_testing(test_data, encoders, feature_names)
        
        if test_features is not None:
            # Compare predictions
            lgb_test_pred = lgb_model.predict(test_features)
            tf_test_pred = tf_model.predict(test_features).flatten()
            
            mse_diff = mean_squared_error(lgb_test_pred, tf_test_pred)
            mae_diff = np.mean(np.abs(lgb_test_pred - tf_test_pred))
            
            print(f"📊 Test Results:")
            print(f"   MSE between models: {mse_diff:.4f}")
            print(f"   MAE between models: {mae_diff:.4f}")
            print(f"   Max difference: {np.max(np.abs(lgb_test_pred - tf_test_pred)):.4f}")
    
    # Save the TensorFlow model
    print("💾 Saving TensorFlow model...")
    tf_model.save("resale_price_estimator_tf_converted")
    
    # Save feature names for reference
    np.save("../models/feature_names.npy", feature_names)
    
    print("✅ Conversion completed!")
    print("📁 Files created:")
    print("   - resale_price_estimator_tf_converted/ (TensorFlow model)")
    print("   - feature_names.npy (Feature names)")
    
    return tf_model, encoders, feature_names

def preprocess_for_testing(data, encoders, feature_names):
    """Preprocess data for testing the converted model"""
    try:
        df = data.copy()
        
        # Apply the same preprocessing as in your app
        bool_cols = ["AuthenticityVerified", "TaxBenefitEligible", "AR_TryOnAvailable", "EcoFriendlyPackaging"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0)
        
        condition_ordinal_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Like New': 4, 'New': 5}
        if 'ConditionGrade' in df.columns:
            df['ConditionGrade'] = df['ConditionGrade'].map(condition_ordinal_map).fillna(2).astype(int)
        
        donation_status_map = {"Available": 0, "Reserved": 1, "Sold": 2}
        if 'DonationStatus' in df.columns:
            df['DonationStatus'] = df['DonationStatus'].map(donation_status_map).fillna(0).astype(int)
        
        categorical_cols = [
            "RetailerName", "Category", "SubCategory", "BrandName", "Size", "Color",
            "StorageLocation", "ShippingPartner", "MaterialType"
        ]
        
        for col in categorical_cols:
            if col in df.columns and col in encoders:
                df[col] = encoders[col].transform(df[col].astype(str))
            elif col in df.columns:
                unique_values = df[col].unique()
                value_to_int = {val: idx for idx, val in enumerate(unique_values)}
                df[col] = df[col].map(value_to_int).fillna(0)
        
        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])
        
        # Remove target column if present
        if 'EstimatedResalePrice' in df.columns:
            df = df.drop(columns=['EstimatedResalePrice'])
        
        # Ensure we have the right number of features
        if len(df.columns) == len(feature_names):
            return df.values
        else:
            print(f"⚠️ Feature count mismatch: {len(df.columns)} vs {len(feature_names)}")
            return None
            
    except Exception as e:
        print(f"❌ Error preprocessing test data: {e}")
        return None

def test_converted_model():
    """Test the converted model with a sample input"""
    print("🧪 Testing converted model...")
    
    try:
        # Load the converted model
        tf_model = tf.keras.models.load_model("resale_price_estimator_tf_converted")
        encoders = joblib.load("../models/label_encoders.pkl")
        feature_names = np.load("../models/feature_names.npy", allow_pickle=True)
        
        # Create sample input
        sample_input = {
            "OriginalPrice": 100.0,
            "RetailerName": "Nike",
            "Category": "Clothing",
            "SubCategory": "Shoes",
            "BrandName": "Nike",
            "Size": "10",
            "Color": "Black",
            "ConditionGrade": "Good",
            "DonationMonth": 1,
            "DonationYear": 2024,
            "StorageLocation": "Warehouse A",
            "AuthenticityVerified": "Yes",
            "DonationStatus": "Available",
            "SustainabilityScore": 0.8,
            "TaxBenefitEligible": "Yes",
            "WinningBidShippingCost": 5.0,
            "ShippingPartner": "FedEx",
            "AR_TryOnAvailable": "No",
            "RewardPoints": 100,
            "MaterialType": "Leather",
            "EcoFriendlyPackaging": "Yes"
        }
        
        # Convert to DataFrame and preprocess
        df = pd.DataFrame([sample_input])
        
        # Apply preprocessing
        bool_cols = ["AuthenticityVerified", "TaxBenefitEligible", "AR_TryOnAvailable", "EcoFriendlyPackaging"]
        for col in bool_cols:
            df[col] = df[col].map({"Yes": 1, "No": 0})
        
        condition_ordinal_map = {'Fair': 1, 'Good': 2, 'Very Good': 3, 'Like New': 4, 'New': 5}
        df['ConditionGrade'] = df['ConditionGrade'].map(condition_ordinal_map).fillna(2).astype(int)
        
        donation_status_map = {"Available": 0, "Reserved": 1, "Sold": 2}
        df['DonationStatus'] = df['DonationStatus'].map(donation_status_map).fillna(0).astype(int)
        
        categorical_cols = [
            "RetailerName", "Category", "SubCategory", "BrandName", "Size", "Color",
            "StorageLocation", "ShippingPartner", "MaterialType"
        ]
        
        for col in categorical_cols:
            if col in encoders and df[col].iloc[0] in encoders[col].classes_:
                df[col] = encoders[col].transform([df[col].iloc[0]])
            else:
                df[col] = 0
        
        # Make prediction
        features = df.to_numpy().astype(np.float32)
        prediction = tf_model.predict(features)[0][0]
        
        print(f"🎯 TensorFlow Model Prediction: ${prediction:.2f}")
        print("✅ Test successful!")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    # Run the conversion
    tf_model, encoders, feature_names = convert_lgbm_to_tensorflow()
    
    # Test the converted model
    test_converted_model() 