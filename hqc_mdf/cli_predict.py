import os
import joblib
import numpy as np
import pandas as pd
from hqc_mdf.data_module.dataset_manager import get_medical_dataset
from hqc_mdf.quantum_module.quantum_model import standard_vqc, advanced_vqc
from hqc_mdf.config.framework_config import QUBITS

def load_inference_artifacts(disease_name):
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
        print("    Please run the training pipeline first.")
        return None
    return artifacts

def get_manual_input(feature_names):
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
    
    artifacts = load_inference_artifacts(disease)
    if not artifacts: return

    feature_names = artifacts['scaler'].feature_names_in_

    print("\nHow would you like to provide patient data?")
    print("1. Enter data manually (Real-time input)")
    print("2. Load an example patient from the dataset")
    choice = input("\nSelect option (1-2): ")

    if choice == "1":
        raw_input = get_manual_input(feature_names)
    else:
        X, _, _ = get_medical_dataset(disease, custom_path=custom_path)
        raw_input = X.iloc[[np.random.randint(0, len(X))]]
        print("\nLoading Random Patient Record:")
        print(raw_input.iloc[0])

    scaled_data = artifacts['scaler'].transform(raw_input)
    pca_data = artifacts['pca'].transform(scaled_data)
    
    if pca_data.shape[1] < QUBITS:
        padding = QUBITS - pca_data.shape[1]
        pca_data = np.hstack([pca_data, np.zeros((1, padding))])

    print("\n" + "-"*30)
    print("   DIAGNOSTIC CONSENSUS")
    print("-"*30)
    
    svm_probs = artifacts['svm'].predict_proba(pca_data)[0]
    rf_probs = artifacts['rf'].predict_proba(pca_data)[0]
    lr_probs = artifacts['lr'].predict_proba(pca_data)[0]
    
    classical_prob = (svm_probs[1] + rf_probs[1] + lr_probs[1]) / 3.0
    classical_pred = 1 if classical_prob >= 0.5 else 0
    classical_confidence = classical_prob if classical_pred == 1 else (1 - classical_prob)
    
    q_std_raw = standard_vqc(pca_data[0], artifacts['q_weights_std'])
    q_adv_raw = advanced_vqc(pca_data[0], artifacts['q_weights_adv'])
    
    q_std_prob = (q_std_raw + 1) / 2.0
    q_adv_prob = (q_adv_raw + 1) / 2.0
    
    quantum_prob = (q_std_prob + q_adv_prob) / 2.0
    q_pred = 1 if quantum_prob >= 0.5 else 0
    quantum_confidence = quantum_prob if q_pred == 1 else (1 - quantum_prob)

    def label(val): return "POSITIVE (Malignant/High Risk)" if val == 1 else "NEGATIVE (Benign/Low Risk)"

    print(f"Classical Models: {label(classical_pred)} (Confidence: {classical_confidence*100:.1f}%)")
    print(f"Quantum Model:    {label(q_pred)} (Confidence: {quantum_confidence*100:.1f}%)")
    
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
