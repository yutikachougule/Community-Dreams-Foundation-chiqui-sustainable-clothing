import requests
import json
import time

# Your deployed service URL
SERVICE_URL = "https://resale-price-estimator-tf-ypo2kd5eyq-uc.a.run.app"

def test_health_endpoint(base_url):
    """Test the health check endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_model_info_endpoint(base_url):
    """Test the model info endpoint"""
    print("📊 Testing model info endpoint...")
    try:
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info: {data['model_type']}")
            print(f"   Feature count: {data['feature_count']}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return False

def test_prediction_endpoint(base_url):
    """Test the prediction endpoint with detailed output"""
    print("🔮 Testing prediction endpoint...")
    
    # Sample input data
    test_data = {
        "OriginalPrice": 100.0,
        "RetailerName": "ChicFusion",
        "Category": "Pants",
        "SubCategory": "Jeans",
        "BrandName": "Randall PLC",
        "Size": "S",
        "Color": "Black",
        "ConditionGrade": "Good",
        "DonationMonth": 1,
        "DonationYear": 2024,
        "StorageLocation": "North Matthew",
        "AuthenticityVerified": "Yes",
        "DonationStatus": "Available",
        "SustainabilityScore": 0.8,
        "TaxBenefitEligible": "Yes",
        "WinningBidShippingCost": 5.0,
        "ShippingPartner": "FedEx",
        "AR_TryOnAvailable": "No",
        "RewardPoints": 100,
        "MaterialType": "Denim",
        "EcoFriendlyPackaging": "Yes",
        "TargetMargin": 10.0
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data)
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print("=" * 50)
            print("📊 RESALE PRICE ESTIMATION RESULTS")
            print("=" * 50)
            print(f"💰 Original Price: ${test_data['OriginalPrice']:.2f}")
            print(f"🎯 Target Profit Margin: {test_data['TargetMargin']}%")
            print(f"📈 Predicted Base Price: ${result['predicted_base_price']:.2f}")
            print(f"💵 Recommended Resale Price: ${result['recommended_resale_price']}")
            print(f"📊 Achieved Profit Margin: {result['achieved_profit_margin']}%")
            print(f"🤖 Model Type: {result['model_type']}")
            print("=" * 50)
            
            # Calculate additional metrics
            profit_amount = result['recommended_resale_price'] - test_data['OriginalPrice']
            print(f"💸 Profit Amount: ${profit_amount:.2f}")
            print(f"📊 Price Increase: {((result['recommended_resale_price'] / test_data['OriginalPrice']) - 1) * 100:.1f}%")
            print("=" * 50)
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_batch_prediction(base_url):
    """Test batch prediction endpoint"""
    print("\n📦 Testing batch prediction endpoint...")
    
    batch_data = {
        "items": [
            {
                "OriginalPrice": 50.0,
                "RetailerName": "Trendify",
                "Category": "Accessories",
                "SubCategory": "Hats",
                "BrandName": "Walters, Murray and Fleming ",
                "Size": "M",
                "Color": "Blue",
                "ConditionGrade": "Very Good",
                "DonationMonth": 2,
                "DonationYear": 2024,
                "StorageLocation": "Port Jerryburgh",
                "AuthenticityVerified": "Yes",
                "DonationStatus": "Available",
                "SustainabilityScore": 0.9,
                "TaxBenefitEligible": "Yes",
                "WinningBidShippingCost": 3.0,
                "ShippingPartner": "UPS",
                "AR_TryOnAvailable": "Yes",
                "RewardPoints": 50,
                "MaterialType": "Silk",
                "EcoFriendlyPackaging": "Yes",
                "TargetMargin": 15.0
            },
            {
                "OriginalPrice": 200.0,
                "RetailerName": "StyleNation",
                "Category": "Shoes",
                "SubCategory": "Boots",
                "BrandName": "Shake Boots",
                "Size": "Standard",
                "Color": "White",
                "ConditionGrade": "Like New",
                "DonationMonth": 3,
                "DonationYear": 2024,
                "StorageLocation": "North Riley",
                "AuthenticityVerified": "Yes",
                "DonationStatus": "Available",
                "SustainabilityScore": 0.7,
                "TaxBenefitEligible": "No",
                "WinningBidShippingCost": 8.0,
                "ShippingPartner": "DHL",
                "AR_TryOnAvailable": "No",
                "RewardPoints": 200,
                "MaterialType": "Leather",
                "EcoFriendlyPackaging": "No",
                "TargetMargin": 20.0
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/batch_predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(batch_data)
        )
        
        if response.status_code == 200:
            results = response.json()["results"]
            print("✅ Batch prediction successful!")
            print("📊 Batch Results:")
            for i, result in enumerate(results, 1):
                if "error" not in result:
                    print(f"   Item {i}: ${result['recommended_resale_price']} (Margin: {result['achieved_profit_margin']}%)")
                else:
                    print(f"   Item {i}: Error - {result['error']}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def test_performance(base_url):
    """Test API performance with multiple requests"""
    print("\n⚡ Testing performance...")
    
    test_data =  {
        "OriginalPrice": 100.0,
        "RetailerName": "ChicFusion",
        "Category": "Pants",
        "SubCategory": "Jeans",
        "BrandName": "Randall PLC",
        "Size": "S",
        "Color": "Black",
        "ConditionGrade": "Good",
        "DonationMonth": 1,
        "DonationYear": 2024,
        "StorageLocation": "North Matthew",
        "AuthenticityVerified": "Yes",
        "DonationStatus": "Available",
        "SustainabilityScore": 0.8,
        "TaxBenefitEligible": "Yes",
        "WinningBidShippingCost": 5.0,
        "ShippingPartner": "FedEx",
        "AR_TryOnAvailable": "No",
        "RewardPoints": 100,
        "MaterialType": "Denim",
        "EcoFriendlyPackaging": "Yes",
        "TargetMargin": 10.0
    }
    
    start_time = time.time()
    successful_requests = 0
    total_requests = 5
    
    for i in range(total_requests):
        try:
            response = requests.post(
                f"{base_url}/predict",
                headers={"Content-Type": "application/json"},
                data=json.dumps(test_data),
                timeout=10
            )
            if response.status_code == 200:
                successful_requests += 1
        except Exception as e:
            print(f"   Request {i+1} failed: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / total_requests
    
    print(f"✅ Performance test completed:")
    print(f"   Successful requests: {successful_requests}/{total_requests}")
    print(f"   Average response time: {avg_time:.2f}s")
    print(f"   Total time: {total_time:.2f}s")
    
    return successful_requests == total_requests

def main():
    """Main test function"""
    print(f"🧪 Testing API at: {SERVICE_URL}")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health_endpoint(SERVICE_URL)),
        ("Model Info", lambda: test_model_info_endpoint(SERVICE_URL)),
        ("Single Prediction", lambda: test_prediction_endpoint(SERVICE_URL)),
        ("Batch Prediction", lambda: test_batch_prediction(SERVICE_URL)),
        ("Performance", lambda: test_performance(SERVICE_URL))
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        if test_func():
            passed += 1
        time.sleep(1)  # Small delay between tests
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your TensorFlow API is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check your deployment.")

if __name__ == "__main__":
    main() 