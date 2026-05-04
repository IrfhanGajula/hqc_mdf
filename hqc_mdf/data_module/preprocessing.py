from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def preprocess_data(X, y):
    """
    Standardizes clinical data and splits into training/testing sets.
    Includes a safety check for small datasets.
    """
    n_samples = len(X)
    
    if n_samples == 0:
        raise ValueError("Diagnostic dataset is empty. Cannot perform clinical preprocessing on 0 samples. Verify your CSV data format.")
    
    if n_samples == 1:
        # Edge case: only 1 sample
        print("Warning: Only 1 sample provided for clinical diagnosis. Using it for both training and testing (Evaluation only).")
        X_train, X_test, y_train, y_test = X, X, y, y
    elif n_samples < 5:
        # Small dataset: force at least one sample into test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        # Standard split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except ValueError as e:
        if "Found array with 0 sample(s)" in str(e):
             raise ValueError("Insufficient clinical samples after dataset splitting. Ensure your dataset has at least 2 valid records.")
        raise e
    
    # Ensure they are dataframes/series for consistency if original were so
    if isinstance(X_train, pd.DataFrame):
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler