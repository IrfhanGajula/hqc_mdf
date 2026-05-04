import os
import joblib
import numpy as np
import pandas as pd
from hqc_mdf.data_module.dataset_manager import get_medical_dataset
from hqc_mdf.quantum_module.quantum_model import standard_vqc, advanced_vqc
from hqc_mdf.config.framework_config import QUBITS

def load_inference_artifacts(disease_name):
    """
    Loads models, scalers, and weights for the specified disease.
    """
    artifacts = {}
    try:
        artifacts['scaler'] = joblib.load(f"models/scaler_{disease_name}.pkl")
        artifacts['pca'] = joblib.load(f"models/pca_{disease_name}.pkl")
        artifacts['svm'] = joblib.load("models/svm.pkl")
        artifacts['rf'] = joblib.load("models/random_forest.pkl")
        artifacts['lr'] = joblib.load("models/logistic_regression.pkl")
        artifacts['q_weights_std'] = np.load("models/quantum_weights_standard_vqc.npy", allow_pickle=True)
        artifacts['q_weights_adv'] = np.load("models/quantum_weights_advanced_vqc.npy", allow_pickle=True)
    except FileNotFoundError as e:
        print(f"\n[!] Error: Required model artifacts not found ({e.filename}).")
        print("    Please run the training pipeline (runner.py) first.")
        return None
    return artifacts

def get_manual_input(feature_names):
    """
    Prompts user for manual feature input with basic validation.
    """
    print(f"\n--- Clinical Data Entry ({len(feature_names)} features) ---")
    data = {}
    for i, feat in enumerate(feature_names):
        while True:
            try:
                val = input(f"[{i+1}/{len(feature_names)}] Enter {feat}: ")
                data[feat] = float(val)
                break
            except ValueError:
                print("    (!) Invalid input. Please enter a numeric value.")
    return pd.DataFrame([data])

def run_prediction(disease="breast_cancer", custom_path=None):
    print("\n" + "="*60)
    print("   HYBRID QUANTUM MEDICAL DIAGNOSTICS: PREDICTION MODE")
    print("="*60)
    
    # 1. Setup
    artifacts = load_inference_artifacts(disease)
    if not artifacts: return

    # Get feature names from the original dataset metadata
    _, _, dataset_md = get_medical_dataset(disease, custom_path=custom_path)
    # We need the actual columns. We can get them from the scaler or by loading 1 row.
    # Scaler.feature_names_in_ is available if it was fitted with a DataFrame
    feature_names = artifacts['scaler'].feature_names_in_

    # 2. Input
    print("\nHow would you like to provide patient data?")
    print("1. Enter data manually (Real-time input)")
    print("2. Load an example patient from the dataset")
    choice = input("\nSelect option (1-2): ")

    if choice == "1":
        raw_input = get_manual_input(feature_names)
    else:
        # Load one example for demo
        X, _, _ = get_medical_dataset(disease, custom_path=custom_path)
        raw_input = X.iloc[[np.random.randint(0, len(X))]]
        print("\nLoading Random Patient Record:")
        print(raw_input.iloc[0])

    # 3. Processing
    # Scale -> PCA
    scaled_data = artifacts['scaler'].transform(raw_input)
    pca_data = artifacts['pca'].transform(scaled_data)
    
    # Ensure shape is (1, QUBITS)
    if pca_data.shape[1] < QUBITS:
        padding = QUBITS - pca_data.shape[1]
        pca_data = np.hstack([pca_data, np.zeros((1, padding))])

    # 4. Hybrid Diagnostics
    print("\n" + "-"*30)
    print("   DIAGNOSTIC CONSENSUS")
    print("-"*30)
    
    # Classical Confidence & Prediction
    svm_probs = artifacts['svm'].predict_proba(pca_data)[0]
    rf_probs = artifacts['rf'].predict_proba(pca_data)[0]
    lr_probs = artifacts['lr'].predict_proba(pca_data)[0]
    
    # Average the probabilities for class 1 across SVM, RF, and LR
    classical_prob = (svm_probs[1] + rf_probs[1] + lr_probs[1]) / 3.0
    classical_pred = 1 if classical_prob >= 0.5 else 0
    classical_confidence = classical_prob if classical_pred == 1 else (1 - classical_prob)
    
    # Quantum Confidence & Prediction
    # run both standard and advanced VQCs
    q_std_raw = standard_vqc(pca_data[0], artifacts['q_weights_std'])
    q_adv_raw = advanced_vqc(pca_data[0], artifacts['q_weights_adv'])
    
    # Map back from [-1, 1] to [0, 1] probability space
    q_std_prob = (q_std_raw + 1) / 2.0
    q_adv_prob = (q_adv_raw + 1) / 2.0
    
    # Average the quantum probabilities
    quantum_prob = (q_std_prob + q_adv_prob) / 2.0
    q_pred = 1 if quantum_prob >= 0.5 else 0
    quantum_confidence = quantum_prob if q_pred == 1 else (1 - quantum_prob)

    def label(val): return "POSITIVE (Malignant/High Risk)" if val == 1 else "NEGATIVE (Benign/Low Risk)"

    print(f"Classical Models: {label(classical_pred)} (Confidence: {classical_confidence*100:.1f}%)")
    print(f"Quantum Model:    {label(q_pred)} (Confidence: {quantum_confidence*100:.1f}%)")
    
    # Conclusion
    final_prob = (classical_prob + quantum_prob) / 2.0
    
    if final_prob >= 0.5:
        decision_label = "POSITIVE (High Priority Follow-up)"
        final_conf = final_prob * 100
        diagnosis_text = "The Hybrid Quantum framework has detected a POSITIVE signature for the health condition. It is recommended that the patient be scheduled for high-priority clinical follow-up."
    else:
        decision_label = "NEGATIVE (Routine Monitoring)"
        final_conf = (1 - final_prob) * 100
        diagnosis_text = "The Hybrid Quantum framework has determined a NEGATIVE signature for the health condition. The consensus indicates low risk, and standard routine monitoring is recommended."
        
    print(f"\nFINAL DECISION: {decision_label} - Overall Confidence: {final_conf:.1f}%")
    
    clinical_summary = (
        f"The classical ML ensemble evaluated the patient with {classical_confidence*100:.1f}% confidence for a {label(classical_pred).split(' ')[0]} diagnosis, "
        f"while the quantum computing variants reached a consensus with {quantum_confidence*100:.1f}% confidence for a {label(q_pred).split(' ')[0]} diagnosis.\n\n"
        f"Combined Conclusion: {diagnosis_text}"
    )

    print("\n[ CLINICAL SUMMARY ]")
    print(clinical_summary)
    print("="*60)

    return {
        "status": decision_label,
        "confidence": final_conf,
        "consensus": f"Classical ({classical_confidence*100:.1f}%) vs Quantum ({quantum_confidence*100:.1f}%)",
        "summary": clinical_summary
    }

if __name__ == "__main__":
    run_prediction()
