#!/bin/bash

# GCP Deployment Script for Resale Price Estimator (TensorFlow)
# This script deploys the converted TensorFlow model to Google Cloud Run

set -e

# Configuration
PROJECT_ID="dynamic-pricing-464100"  # Replace with your GCP project ID
SERVICE_NAME="resale-price-estimator-tf"
REGION="us-central1"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo "🚀 Starting GCP deployment for Resale Price Estimator (TensorFlow)..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install it first:"
    echo "https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if model files exist
if [ ! -d "../resale_price_estimator_tf_converted" ]; then
    echo "❌ TensorFlow model not found. Please run the conversion script first:"
    echo "python ../convert_lgbm_simple.py"
    exit 1
fi

if [ ! -f "../label_encoders.pkl" ]; then
    echo "❌ Label encoders not found. Please ensure label_encoders.pkl exists."
    exit 1
fi

if [ ! -f "../feature_names.npy" ]; then
    echo "❌ Feature names not found. Please ensure feature_names.npy exists."
    exit 1
fi

# Set the project
echo "📋 Setting GCP project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Copy model files to deployment directory
echo "📁 Copying model files..."
cp -r ../resale_price_estimator_tf_converted/ .
cp ../label_encoders.pkl .
cp ../feature_names.npy .

# Build and push the Docker image
echo "🏗️ Building Docker image..."
docker build -t $IMAGE_NAME .

echo "📤 Pushing image to Google Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --concurrency 80 \
    --max-instances 10 \
    --set-env-vars "MODEL_PATH=resale_price_estimator_tf_converted"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')

echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "📝 API Endpoints:"
echo "  Health Check: $SERVICE_URL/health"
echo "  Model Info: $SERVICE_URL/model_info"
echo "  Prediction: $SERVICE_URL/predict (POST)"
echo "  Batch Prediction: $SERVICE_URL/batch_predict (POST)"
echo ""
echo "🧪 Test the API with:"
echo "curl -X POST $SERVICE_URL/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{"
echo "    \"OriginalPrice\": 100.0,"
echo "    \"RetailerName\": \"Trendify\","
echo "    \"Category\": \"Pants\","
echo "    \"SubCategory\": \"Jeans\","
echo "    \"BrandName\": \"Randall PLC\","
echo "    \"Size\": \"S\","
echo "    \"Color\": \"Black\","
echo "    \"ConditionGrade\": \"Good\","
echo "    \"DonationMonth\": 1,"
echo "    \"DonationYear\": 2024,"
echo "    \"StorageLocation\": \"North Matthew\","
echo "    \"AuthenticityVerified\": \"Yes\","
echo "    \"DonationStatus\": \"Available\","
echo "    \"SustainabilityScore\": 0.8,"
echo "    \"TaxBenefitEligible\": \"Yes\","
echo "    \"WinningBidShippingCost\": 5.0,"
echo "    \"ShippingPartner\": \"FedEx\","
echo "    \"AR_TryOnAvailable\": \"No\","
echo "    \"RewardPoints\": 100,"
echo "    \"MaterialType\": \"Denim\","
echo "    \"EcoFriendlyPackaging\": \"Yes\","
echo "    \"TargetMargin\": 10.0"
echo "  }'"

# Clean up copied files
echo "🧹 Cleaning up temporary files..."
rm -rf resale_price_estimator_tf_converted/
rm -f label_encoders.pkl
rm -f feature_names.npy 