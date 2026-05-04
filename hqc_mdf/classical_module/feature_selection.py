from sklearn.decomposition import PCA
import numpy as np

def apply_pca(X_train, X_test, n_components=6):
    """
    Applies PCA with a safety check for n_components and zero-pads the output 
    to ensure the resulting feature vector always has length n_components.
    """
    n_samples, n_features = X_train.shape
    
    # PCA requires n_components <= min(n_samples, n_features)
    # Also n_components must be <= n_samples - 1 if svd_solver is 'full' or 'auto'
    n_comp_safe = min(n_components, n_samples - 1, n_features)
    
    if n_comp_safe < 1:
        # Fallback for extremely small datasets: just take what we have or return zeros if empty
        print(f"Warning: Dataset too small for PCA. Using raw features (padded).")
        X_train_pca = X_train
        X_test_pca = X_test
    else:
        pca = PCA(n_components=n_comp_safe)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
    
    # Padding logic: If we have fewer than n_components, pad with zeros
    current_cols = X_train_pca.shape[1]
    if current_cols < n_components:
        padding_needed = n_components - current_cols
        X_train_pca = np.hstack([X_train_pca, np.zeros((X_train_pca.shape[0], padding_needed))])
        X_test_pca = np.hstack([X_test_pca, np.zeros((X_test_pca.shape[0], padding_needed))])
        
    pca_obj = pca if n_comp_safe >= 1 else None
    return X_train_pca, X_test_pca, pca_obj
