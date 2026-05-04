import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score
from pennylane import numpy as np
import time
import json
import os

class ModelComparator:
    """
    Data-First Diagnostic Reporting Engine.
    Generates CLI dashboards and raw JSON manifests for absolute transparency.
    """
    def __init__(self, dataset_md=None):
        self.results = {}
        self.predictions = {} # Store raw labels for output comparison
        self.dataset_md = dataset_md
        self.analysis_notes = ""

    def add_classical_results(self, detailed_results):
        for name, metrics in detailed_results.items():
            self.results[name] = {
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "Time": metrics["Execution Time"],
                "Efficiency": metrics["Accuracy"] / (metrics["Execution Time"] + 1e-6),
                "Category": "CPU Execution",
                "Type": "Classical"
            }
            self.predictions[name] = metrics["Predictions"]

    def evaluate_quantum_model(self, name, X_test, y_test, weights, circuit_fn, training_md):
        print(f"Evaluating {name}...")
        start_time = time.time()
        predictions = []
        for f in X_test:
            raw_pred = circuit_fn(f, weights)
            pred = 1 if raw_pred > 0 else 0
            predictions.append(pred)
        end_time = time.time()
        
        eval_time = end_time - start_time
        total_time = training_md["training_time"] + eval_time
        
        self.results[name] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "Time": total_time,
            "Efficiency": accuracy_score(y_test, predictions) / (total_time + 1e-6),
            "Category": "Quantum Simulation",
            "Type": "Quantum",
            "MD": training_md
        }
        self.predictions[name] = predictions

    def save_json_manifest(self):
        """
        Saves all raw variables to a JSON file for manual audit.
        """
        manifest = {
            "metadata": self.dataset_md,
            "timestamp": time.ctime(),
            "metrics": self.results,
            "conclusion": self.analysis_notes
        }
        
        # Remove numpy types for JSON serialization
        def clean_dict(d):
            new_dict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    new_dict[k] = clean_dict(v)
                elif hasattr(v, 'item'): # numpy types
                    new_dict[k] = v.item()
                else:
                    new_dict[k] = v
            return new_dict

        clean_manifest = clean_dict(manifest)
        
        path = "diagnostic_results.json"
        with open(path, "w") as f:
            json.dump(clean_manifest, f, indent=4)
        
        print(f"\n[DATA AUDIT] Raw results saved to: {path}")

    def print_cli_dashboard(self):
        """
        Prints a high-visibility data dashboard to the terminal.
        """
        df = pd.DataFrame(self.results).T.reset_index()
        df.rename(columns={'index': 'Model'}, inplace=True)
        
        print("\n" + "="*70)
        print("          HQC-MDF CLINICAL DIAGNOSTIC DASHBOARD (CLI MODE)")
        print("="*70)
        
        print(f"\n[DATASET PROFILE]")
        print(f" Source:    {self.dataset_md['name']}")
        print(f" Samples:   {self.dataset_md['total_samples']}")
        print(f" Features:  {self.dataset_md['features']} Dimensions")
        print(f" Domain:    {self.dataset_md.get('domain', 'Medical/Clinical')}")

        print(f"\n[PERFORMANCE BENCHMARKS]")
        # Custom formatted table
        header = f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'Time (s)':<10}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        
        for _, row in df.iterrows():
            print(f"{row['Model']:<20} | {row['Accuracy']:<10.4f} | {row['Precision']:<10.4f} | {row['Recall']:<10.4f} | {row['Time']:<10.4f}")
        print("-" * len(header))

        # Agreement Analysis
        classical_models = [n for n, m in self.results.items() if m['Type'] == 'Classical']
        quantum_models = [n for n, m in self.results.items() if m['Type'] == 'Quantum']
        
        if classical_models and quantum_models:
            best_c = max(classical_models, key=lambda x: self.results[x]['Accuracy'])
            best_q = max(quantum_models, key=lambda x: self.results[x]['Accuracy'])
            
            preds_c = np.array(self.predictions[best_c])
            preds_q = np.array(self.predictions[best_q])
            agreement = (preds_c == preds_q).mean() * 100
            
            print(f"\n[CLINICAL CONSENSUS]")
            print(f" Top Models: {best_c} vs {best_q}")
            print(f" Agreement:  {agreement:.2f}% of diagnostic cases match directly.")

        print(f"\n[AUTO-GENERATED INSIGHTS]")
        print(self.analysis_notes)
        print("\n" + "="*70 + "\n")

    def generate_plots(self):
        """
        Saves static visuals to the models folder for archiving.
        """
        os.makedirs("models/plots", exist_ok=True)
        df = pd.DataFrame(self.results).T.reset_index()
        df.rename(columns={'index': 'Model'}, inplace=True)
        
        # 1. Accuracy Matrix
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Accuracy', hue='Type', data=df, palette='viridis')
        plt.title("Diagnostic Accuracy Matrix")
        plt.savefig("models/plots/accuracy_benchmarks.png")
        plt.close()

        # 2. Clinical Reliability (Precision/Recall)
        plt.figure(figsize=(10, 6))
        metrics_melted = df.melt(id_vars='Model', value_vars=['Precision', 'Recall'], var_name='Metric', value_name='Score')
        sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_melted, palette='magma')
        plt.title("Medical Reliability (Precision/Recall)")
        plt.savefig("models/plots/clinical_reliability.png")
        plt.close()

        # 3. Execution Overhead
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Model', y='Time', hue='Category', data=df, palette='coolwarm')
        plt.yscale('log')
        plt.title("Computational Overhead (Execution Time)")
        plt.savefig("models/plots/execution_overhead.png")
        plt.close()
        
        print(f"[VISUALS] All clinical visuals saved to: models/plots/")
