# Streamlit Resale Price Estimator

A web application for estimating resale prices of products using machine learning.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Open your browser and navigate to the provided URL (usually http://localhost:8501)
2. Fill in the product details in the form
3. Click "Predict Resale Price" to get the estimation
4. View the detailed breakdown of the prediction

## Features

- Interactive web interface
- Real-time price predictions
- Detailed breakdown of pricing factors
- Support for various product categories
- Profit margin calculations

## Model

This application uses a LightGBM model trained on historical resale data to predict optimal resale prices. 