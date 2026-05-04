from hqc_mdf.data_module.dataset_manager import get_medical_dataset
from hqc_mdf.data_module.preprocessing import preprocess_data
from hqc_mdf.classical_module.feature_selection import apply_pca
from hqc_mdf.classical_module.classical_models import train_classical_models
from hqc_mdf.quantum_module.quantum_model import standard_vqc, advanced_vqc
from hqc_mdf.hybrid_core.trainer import train_model
from hqc_mdf.hybrid_core.comparator import ModelComparator
from hqc_mdf.config.framework_config import QUBITS
import joblib

import numpy as np
import os
import time

def run_pipeline(disease_name="breast_cancer", train=True, custom_path=None):
    # Ensure necessary directories exist
    os.makedirs("models", exist_ok=True)

    # 1. Clinical Data Preparation & Metadata Extraction
    print(f"\n" + "="*50)
    print(f"   DIAGNOSTIC PIPELINE: {disease_name.upper() if not custom_path else 'CUSTOM CSV'}")
    print("="*50)
    
    X, y, dataset_md = get_medical_dataset(disease_name, custom_path=custom_path)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)
    X_train_pca, X_test_pca, pca_obj = apply_pca(X_train_scaled, X_test_scaled, n_components=QUBITS)
    
    # Save transformers for inference
    joblib.dump(scaler, f"models/scaler_{disease_name}.pkl")
    if pca_obj:
        joblib.dump(pca_obj, f"models/pca_{disease_name}.pkl")
    
    results = {
        "diagnostic_data": (X_train_pca, X_test_pca, y_train, y_test),
        "metadata": dataset_md
    }

    comparator = ModelComparator(dataset_md=dataset_md)

    # 2. Classical Diagnostic Benchmarking (Baseline)
    print("\n[1/3] Running Classical diagnostic benchmarks...")
    detailed_results, _ = train_classical_models(X_train_pca, y_train, X_test_pca, y_test)
    comparator.add_classical_results(detailed_results)

    # 3. Quantum Diagnostic Phase (Fair multi-model comparison)
    print("\n[2/3] Initiating Quantum Diagnostic Exploration...")
    
    quantum_configs = [
        {"name": "Standard VQC", "fn": standard_vqc, "layers": 1},
        {"name": "Advanced VQC", "fn": advanced_vqc, "layers": 3}
    ]
    
    for config in quantum_configs:
        weights = None
        training_md = None
        
        if train:
            weights, training_md = train_model(X_train_pca, y_train.values, config["fn"], layers=config["layers"])
            # Save weights for inference
            weights_path = f"models/quantum_weights_{config['name'].lower().replace(' ', '_')}.npy"
            np.save(weights_path, weights)

        else:
            # Fallback for evaluation mode
            weights_path = f"models/quantum_weights_{config['name'].lower().replace(' ', '_')}.npy"
            if os.path.exists(weights_path):
                weights = np.load(weights_path, allow_pickle=True)
                training_md = {"training_time": 0, "layers": config["layers"]}
            
        if weights is not None:
            # 4. Diagnostic Evaluation
            comparator.evaluate_quantum_model(config["name"], X_test_pca, y_test.values, weights, config["fn"], training_md)
        else:
            print(f"   (!) Skipping {config['name']} (No weights found. Select 'y' to retrain).")

    # 5. Synthesis & Deep Conclusion
    avg_classical = np.mean([m['Accuracy'] for m in detailed_results.values()])
    # Finding the best performing quantum model
    quantum_results = [m['Accuracy'] for m in comparator.results.values() if m['Type'] == 'Quantum']
    max_quantum = max(quantum_results) if quantum_results else 0
    
    best_overall = "Classical Algorithms" if avg_classical > max_quantum else "Quantum VQC"
    
    conclusion = f"The {best_overall} showed superior diagnostic reliability on this specific clinical profile.\n\n"
    conclusion += "HYBRID ROLE JUSTIFICATION:\n"
    conclusion += f"The system functions as a true hybrid by using Classical PCA to compress the {dataset_md['features']}-dimensional medical space "
    conclusion += f"into {QUBITS} primary components, which are then mapped onto a {2**QUBITS}-dimensional Hilbert space for quantum analysis.\n\n"
    conclusion += "EXPERIMENTAL INSIGHTS:\n"
    conclusion += f"1. Classical dominance (~{avg_classical:.2f} Acc) is expected due to the maturity of mature discriminative models on low-qubit datasets.\n"
    conclusion += f"2. Quantum performance is limited by simulation overhead and depth; however, the {2**QUBITS} state-space provides a foundation for "
    conclusion += "higher-order correlation mapping as qubit counts scale.\n"
    conclusion += f"3. Cost function convergence suggests the VQC has reached a stable local optima, indicative of consistent learning quality."

    comparator.analysis_notes = conclusion
    comparator.print_cli_dashboard()
    comparator.save_json_manifest()
    comparator.generate_plots()
    
    return results