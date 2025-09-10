# Dynamic Pricing - Resale Price Estimator

A machine learning project for estimating optimal resale prices of products using LightGBM and TensorFlow models.

## Project Structure

```
Dynamic Pricing/
├── data/                          # Data files
│   ├── feature_matrix.csv
│   ├── processed_dataset.csv
│   └── sample_product_dataset.csv
├── models/                        # Trained models and artifacts
│   ├── lgbm_resale_model.pkl
│   ├── lgbm_resale_model.joblib
│   ├── xgb_resale_model.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.npy
│   └── resale_price_estimator_tf_converted/
├── scripts/                       # Training and utility scripts
│   ├── train.py
│   ├── train_xgb.py
│   ├── feature_engineering.py
│   ├── feature_matrix.py
│   ├── convert_lgbm_simple.py
│   ├── test.py
│   └── requirements_conversion.txt
├── gcp_deployment/                # Google Cloud Platform deployment
│   ├── main.py                    # Flask API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── test_api.py
│   ├── deploy_clean.ps1
│   └── README.md
└── streamlit_deployment/          # Streamlit web application
    ├── app.py                     # Streamlit app
    ├── requirements.txt
    └── README.md
```

## Deployment Options

### 1. Google Cloud Platform (GCP) Deployment
- **Location**: `gcp_deployment/`
- **Type**: REST API using Flask
- **Features**: 
  - Scalable cloud deployment
  - RESTful API endpoints
  - Docker containerization
  - Auto-scaling capabilities

### 2. Streamlit Web Application
- **Location**: `streamlit_deployment/`
- **Type**: Interactive web interface
- **Features**:
  - User-friendly web interface
  - Real-time predictions
  - Interactive forms
  - Local development

## Quick Start

### For GCP Deployment:
```bash
cd gcp_deployment
./deploy_clean.ps1  # On Windows
```

### For Streamlit App:
```bash
cd streamlit_deployment
pip install -r requirements.txt
streamlit run app.py
```

## Models

- **LightGBM**: Primary model for resale price prediction
- **TensorFlow**: Converted model for cloud deployment
- **XGBoost**: Alternative model (experimental)

## Features

- Product price estimation
- Profit margin calculations
- Multiple deployment options
- Scalable cloud infrastructure
- Interactive web interface 