# GCP Deployment for Resale Price Estimator (TensorFlow)

This folder contains all the necessary files to deploy your converted TensorFlow model to Google Cloud Platform using Cloud Run.

## 📁 Folder Structure

```
deployment/
├── main.py                 # Flask API server
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── .dockerignore          # Files to exclude from Docker build
├── deploy.sh              # Deployment automation script
├── test_api.py            # API testing script
└── README.md              # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud CLI (gcloud)** installed and configured
3. **Docker** installed on your machine
4. **Python 3.11+** installed

### Step 1: Convert Your Model

Before deploying, you need to convert your LightGBM model to TensorFlow:

```bash
# Navigate to the parent directory
cd ..

# Run the conversion script
python convert_lgbm_simple.py
```

This will create:
- `resale_price_estimator_tf_converted/` - TensorFlow model
- `label_encoders.pkl` - Label encoders
- `feature_names.npy` - Feature names

### Step 2: Configure Deployment

Edit the `deploy.sh` file and replace `your-gcp-project-id` with your actual GCP project ID:

```bash
PROJECT_ID="your-actual-project-id"
```

### Step 3: Deploy

```bash
# Navigate to deployment directory
cd deployment

# Make the script executable (on Windows, run in Git Bash)
chmod +x deploy.sh

# Run the deployment
./deploy.sh
```

## 📊 API Endpoints

### Health Check
- **URL**: `GET /health`
- **Response**: Model status and configuration

### Model Information
- **URL**: `GET /model_info`
- **Response**: Model type, feature count, and feature names

### Single Prediction
- **URL**: `POST /predict`
- **Content-Type**: `application/json`
- **Request Body**: Product data (see example below)
- **Response**: Prediction results with pricing and margin information

### Batch Prediction
- **URL**: `POST /batch_predict`
- **Content-Type**: `application/json`
- **Request Body**: `{"items": [product1, product2, ...]}`
- **Response**: Array of prediction results

## 🧪 Testing Your Deployment

### Using the Test Script

```bash
python test_api.py
```

### Manual Testing

```bash
# Health check
curl https://your-service-url/health

# Model info
curl https://your-service-url/model_info

# Single prediction
curl -X POST https://your-service-url/predict \
  -H 'Content-Type: application/json' \
  -d '{
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
    "EcoFriendlyPackaging": "Yes",
    "TargetMargin": 10.0
  }'
```

## 📈 Expected Response Format

```json
{
  "predicted_base_price": 85.50,
  "recommended_resale_price": 110.99,
  "achieved_profit_margin": 10.99,
  "target_margin": 10.0,
  "original_price": 100.0,
  "model_type": "TensorFlow (converted from LightGBM)"
}
```

## 🔧 Configuration

### Cloud Run Settings

The deployment uses these optimized settings:

- **Memory**: 2Gi (sufficient for TensorFlow inference)
- **CPU**: 1 (adequate for most workloads)
- **Max Instances**: 10 (prevents cost spikes)
- **Concurrency**: 80 (handles multiple requests per instance)
- **Timeout**: 300 seconds (for complex predictions)

### Cost Optimization

- **Scales to zero** when not in use
- **Auto-scaling** based on demand
- **Max instances limit** prevents cost spikes

## 🔒 Security

- **HTTPS** automatically enabled
- **Non-root user** in container
- **Isolated environment** in Google's infrastructure
- **No authentication** (for testing - add for production)

## 🚨 Troubleshooting

### Common Issues

1. **Model Files Missing**
   ```bash
   # Ensure conversion was successful
   ls -la ../resale_price_estimator_tf_converted/
   ls -la ../label_encoders.pkl
   ls -la ../feature_names.npy
   ```

2. **Build Failures**
   ```bash
   # Check Docker build logs
   docker build -t test-image . --progress=plain
   ```

3. **Memory Issues**
   ```bash
   # Increase memory allocation
   gcloud run services update resale-price-estimator-tf \
     --memory 4Gi --region=us-central1
   ```

### Debug Commands

```bash
# View service logs
gcloud logs read --service=resale-price-estimator-tf --limit=100

# Check service status
gcloud run services describe resale-price-estimator-tf --region=us-central1

# Test locally
docker run -p 8080:8080 gcr.io/YOUR_PROJECT/resale-price-estimator-tf
```

## 📊 Monitoring

### View Logs
```bash
gcloud logs read --service=resale-price-estimator-tf --limit=50
```

### Monitor Performance
- Go to [Google Cloud Console](https://console.cloud.google.com/run)
- Select your service
- View metrics and logs

## 🔄 Updates and Maintenance

### Update Model
1. Run the conversion script again
2. Rebuild and redeploy:
   ```bash
   ./deploy.sh
   ```

### Update Dependencies
1. Update `requirements.txt`
2. Rebuild and redeploy:
   ```bash
   ./deploy.sh
   ```

### Rollback
```bash
# List revisions
gcloud run revisions list --service=resale-price-estimator-tf --region=us-central1

# Rollback to previous revision
gcloud run services update-traffic resale-price-estimator-tf \
  --to-revisions=REVISION_NAME=100 --region=us-central1
```

## 💰 Cost Estimation

### Typical Costs (US Central)
- **Idle time**: $0.00 (scales to zero)
- **Active time**: ~$0.00002400 per 100ms
- **Memory**: ~$0.00000250 per GB-second
- **CPU**: ~$0.00002400 per vCPU-second

### Monthly Estimate
- **Low usage** (1000 requests/day): ~$5-10/month
- **Medium usage** (10000 requests/day): ~$20-50/month
- **High usage** (100000 requests/day): ~$100-200/month

## 📞 Support

For issues with:
- **GCP Services**: Check [Google Cloud Documentation](https://cloud.google.com/docs)
- **Cloud Run**: Visit [Cloud Run Documentation](https://cloud.google.com/run/docs)
- **Model Issues**: Check the application logs and test locally first

---

**Happy Deploying! 🚀** 