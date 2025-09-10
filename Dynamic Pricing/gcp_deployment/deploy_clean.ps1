# GCP Deployment Script for Resale Price Estimator (TensorFlow)
# Clean PowerShell version for Windows

# Configuration
$PROJECT_ID = "dynamic-pricing-464100"
$SERVICE_NAME = "resale-price-estimator-tf"
$REGION = "us-central1"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/$SERVICE_NAME"

Write-Host "Starting GCP deployment for Resale Price Estimator (TensorFlow)..." -ForegroundColor Green

# Check if gcloud is installed
try {
    gcloud --version | Out-Null
    Write-Host "gcloud CLI found" -ForegroundColor Green
} catch {
    Write-Host "gcloud CLI is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Check if docker is installed
try {
    docker --version | Out-Null
    Write-Host "Docker found" -ForegroundColor Green
} catch {
    Write-Host "Docker is not installed. Please install it first:" -ForegroundColor Red
    Write-Host "https://docs.docker.com/get-docker/" -ForegroundColor Yellow
    exit 1
}

# Check if model files exist
if (-not (Test-Path "../resale_price_estimator_tf_converted")) {
    Write-Host "TensorFlow model not found. Please run the conversion script first:" -ForegroundColor Red
    Write-Host "python ../convert_lgbm_simple.py" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "../label_encoders.pkl")) {
    Write-Host "Label encoders not found. Please ensure label_encoders.pkl exists." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "../feature_names.npy")) {
    Write-Host "Feature names not found. Please ensure feature_names.npy exists." -ForegroundColor Red
    exit 1
}

# Set the project
Write-Host "Setting GCP project to: $PROJECT_ID" -ForegroundColor Cyan
gcloud config set project $PROJECT_ID

# Enable required APIs
Write-Host "Enabling required APIs..." -ForegroundColor Cyan
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Copy model files to deployment directory
Write-Host "Copying model files..." -ForegroundColor Cyan
Copy-Item -Recurse "../resale_price_estimator_tf_converted/" .
Copy-Item "../label_encoders.pkl" .
Copy-Item "../feature_names.npy" .

# Build and push the Docker image
Write-Host "Building Docker image..." -ForegroundColor Cyan
docker build -t $IMAGE_NAME .

Write-Host "Pushing image to Google Container Registry..." -ForegroundColor Cyan
docker push $IMAGE_NAME

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..." -ForegroundColor Cyan
gcloud run deploy $SERVICE_NAME `
    --image $IMAGE_NAME `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 1 `
    --timeout 300 `
    --concurrency 80 `
    --max-instances 10 `
    --set-env-vars "MODEL_PATH=resale_price_estimator_tf_converted"

# Get the service URL
$SERVICE_URL = gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'

Write-Host "Deployment completed successfully!" -ForegroundColor Green
Write-Host "Service URL: $SERVICE_URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor Cyan
Write-Host "  Health Check: $SERVICE_URL/health" -ForegroundColor White
Write-Host "  Model Info: $SERVICE_URL/model_info" -ForegroundColor White
Write-Host "  Prediction: $SERVICE_URL/predict (POST)" -ForegroundColor White
Write-Host "  Batch Prediction: $SERVICE_URL/batch_predict (POST)" -ForegroundColor White
Write-Host ""
Write-Host "Test the API with the test script:" -ForegroundColor Cyan
Write-Host "python test_api.py" -ForegroundColor White

# Clean up copied files
Write-Host "Cleaning up temporary files..." -ForegroundColor Cyan
Remove-Item -Recurse -Force "resale_price_estimator_tf_converted/"
Remove-Item "label_encoders.pkl"
Remove-Item "feature_names.npy"

Write-Host ""
Write-Host "Deployment complete! Your service is now live at: $SERVICE_URL" -ForegroundColor Green 