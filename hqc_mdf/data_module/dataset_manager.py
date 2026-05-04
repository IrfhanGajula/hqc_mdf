from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

def detect_domain(X, df_columns, file_name):
    """
    Heuristic to detect if a dataset is medical/clinical.
    """
    medical_keywords = {
        'age', 'blood', 'heart', 'pressure', 'glucose', 'insulin', 'bmi', 
        'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 
        'concavity', 'symmetry', 'fractal', 'diagnosis', 'patient', 'clinical', 
        'medical', 'disease', 'symptom', 'treatment', 'cancer', 'diabetes', 
        'cholesterol', 'survival', 'mortal', 'death', 'biopsy', 'test', 'score'
    }
    
    # Check column names
    found_keywords = [col for col in df_columns if any(key in str(col).lower() for key in medical_keywords)]
    
    # Check file name
    in_file_name = any(key in file_name.lower() for key in medical_keywords)
    
    if not found_keywords and not in_file_name:
        print("\n" + "!"*60)
        print("   WARNING: OUT-OF-DOMAIN DATASET DETECTED")
        print("   The provided data does not seem to contain clinical or medical features.")
        print("   The Hybrid Quantum Framework is optimized for medical diagnostics.")
        print("!"*60 + "\n")
        return "Out-of-Domain"
    return "Clinical/Medical"

def load_custom_clinical_csv(file_path, target_column=None):
    """
    Loads a custom medical CSV file and splits it into Features (X) and Labels (y).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Clinical data file not found at: {file_path}")
        
    df = pd.read_csv(file_path)
    df = df.dropna(axis=1, how='all')
    
    # Metadata initialization
    md = {
        "name": os.path.basename(file_path),
        "total_samples": len(df)
    }
    
    id_cols = [col for col in df.columns if str(col).lower() in ['id', 'patient_id', 'case_id', '0']]
    if id_cols:
        df = df.drop(columns=id_cols)
    
    # Domain Detection
    domain = detect_domain(df, df.columns, os.path.basename(file_path))
    md["domain"] = domain

    diagnosis_keywords = ['diagnosis', 'result', 'target', 'label', 'status', 'class', 'disease', 'outcome']
    if target_column is None:
        for col in df.columns:
            if any(key in str(col).lower() for key in diagnosis_keywords):
                target_column = col
                break
        if target_column is None and 'glyhb' in df.columns:
            df['Diagnosis_Derived'] = (df['glyhb'] >= 6.5).astype(int)
            target_column = 'Diagnosis_Derived'
        if target_column is None:
            target_column = df.columns[-1]

    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Advanced Label Encoding
    if y.dtype == 'object':
        mapping = {
            'M': 1, 'B': 0, 'MALIGNANT': 1, 'BENIGN': 0,
            '1': 1, '0': 0, 'POSITIVE': 1, 'NEGATIVE': 0,
            'YES': 1, 'NO': 0, 'TRUE': 1, 'FALSE': 0,
            'HIGH': 1, 'LOW': 0, 'NORMAL': 0, 'ABNORMAL': 1
        }
        
        # Try mapped conversion
        y_temp = y.str.strip().str.upper().map(mapping)
        
        # If mapping didn't work (too many NaNs), fallback to factorization
        if y_temp.isna().sum() > len(y) * 0.5:
            print(f"INFO: Standard medical mapping failed. Using automatic clinical factorization for '{target_column}'.")
            y_encoded, uniques = pd.factorize(y)
            y = pd.Series(y_encoded, index=y.index)
            # Update metadata with classes
            md["classes_map"] = {i: val for i, val in enumerate(uniques)}
        else:
            y = y_temp
    
    # Encode categorical features in X
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = pd.factorize(X[col])[0]

    mask = y.notna()
    X = X[mask].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y[mask].astype(int)
    
    if len(X) == 0:
        raise ValueError(f"Dataset parsing failed: 0 valid samples found after clinical preprocessing. Check column '{target_column}' values.")
    
    md["features"] = len(X.columns)
    md["classes"] = len(y.unique())
    md["problem_type"] = "Binary Classification"
    
    print(f"Successfully loaded {len(X)} records for analysis.")
    return X, y, md

def get_medical_dataset(disease_name="breast_cancer", custom_path=None):
    """
    Fetches a medical dataset for quantum diagnostic classification.
    """
    if custom_path:
        return load_custom_clinical_csv(custom_path)
        
    if disease_name == "breast_cancer":
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        md = {
            "name": "Wisconsin Breast Cancer",
            "total_samples": len(X),
            "features": len(X.columns),
            "classes": 2,
            "problem_type": "Binary Classification"
        }
        return X, y, md
    else:
        raise ValueError(f"Medical dataset {disease_name} not currently supported. Please provide a CSV path.")
