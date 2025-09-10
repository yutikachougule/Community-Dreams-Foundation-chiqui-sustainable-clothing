import pandas as pd

def display_feature_matrix(processed_csv, target_column="EstimatedResalePrice", output_csv="../data/feature_matrix.csv"):
    # Load the processed dataset
    df = pd.read_csv(processed_csv)

    drop_cols = ['DonationDate','ProductID', 'ProductName', 'ProductImageURL']
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    # Create the feature matrix by dropping the target column
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
    else:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")

    # Print basic info
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Feature names: {list(X.columns)}\n")

    X.to_csv(output_csv, index=False)

    

if __name__ == "__main__":
    # Run with your processed dataset
    display_feature_matrix("../data/sample_product_dataset.csv")
