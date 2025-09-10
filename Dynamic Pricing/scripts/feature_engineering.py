import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def feature_engineering(input_csv, output_csv="../data/processed_dataset.csv", save_encoders=True):
    df = pd.read_csv(input_csv)

    # Handle Boolean-like Columns
    bool_cols = ['AuthenticityVerified', 'TaxBenefitEligible', 'AR_TryOnAvailable', 'EcoFriendlyPackaging']
    for col in bool_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0}).fillna(0).astype(int)

    df['DonationStatus'] = df['DonationStatus'].map({"Available": 0,"Reserved": 1,"Sold": 2}).fillna(0).astype(int)


    # Handle DonationDate
    df['DonationDate'] = pd.to_datetime(df['DonationDate'], errors='coerce')
    df['DonationMonth'] = df['DonationDate'].dt.month.fillna(0).astype(int)
    df['DonationYear'] = df['DonationDate'].dt.year.fillna(0).astype(int)

    # Encode ConditionGrade (Ordinal: 1–5)
    df['ConditionGrade'] = df['ConditionGrade'].str.strip().str.title()
    condition_ordinal_map = {
        'Fair': 1,
        'Good': 2,
        'Very Good': 3,
        'Like New': 4,
        'New': 5
    }
    df['ConditionGrade'] = df['ConditionGrade'].map(condition_ordinal_map).fillna(2).astype(int)

    # Fill other missing values
    df.fillna({'Color': 'Unknown'}, inplace=True)

    # Encode Other Categorical Columns
    label_cols = ['RetailerName', 'Category', 'SubCategory', 'BrandName',
                  'Size', 'Color', 'StorageLocation', 'ShippingPartner', 'MaterialType']
    
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Save Processed Dataset
    df.to_csv(output_csv, index=False)
    print(f"Feature engineered data saved to: {output_csv}")

    # Save Encoders
    if save_encoders:
        joblib.dump(encoders, "../models/label_encoders.pkl")
        print("Label encoders saved to: label_encoders.pkl")

if __name__ == "__main__":
    feature_engineering("../data/sample_product_dataset.csv")
