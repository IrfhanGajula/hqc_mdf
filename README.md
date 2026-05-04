# Hybrid Quantum Medical Diagnostics Framework (HQC-MDF)

> **One-Line Summary**: "User uploads a dataset once, the system processes it through classical and quantum models, and outputs a comparison with a final performance analysis."

## 🔹 Project Overview
This project builds a **Hybrid Classical–Quantum Analytics System** that takes a dataset, processes it using classical machine learning techniques, sends it to a quantum model for advanced computation, and finally compares both results to understand which performs better.

## 🔹 How the System Works (End-to-End Flow)
1.  **Dataset Upload**: User provides a medical dataset (CSV format).
2.  **Data Processing**: System cleans, scales, and prepares clinical data.
3.  **Dimensionality Reduction**: Automated PCA reduces data size for quantum hardware compatibility.
4.  **Classical Model Execution**: Runs traditional ML benchmarks (SVM, Random Forest, Logistic Regression).
5.  **Quantum Model Execution**: Encodes data and executes a Variational Quantum Circuit (VQC).
6.  **Comparison & Evaluation**: Evaluates models based on accuracy, time, and efficiency.
7.  **Final Output Generation**: Produces comparisons and a definitive clinical conclusion.

## 📂 Project Structure
- **`hybrid_core/`**: Orchestration, training logic, and model comparison.
- **`quantum_module/`**: VQC architecture and quantum feature encoding.
- **`classical_module/`**: Baseline ML models and feature selection (PCA).
- **`data_module/`**: Medical dataset management and clinical preprocessing.
- **`config/`**: Hyperparameters (Qubits, Learning Rate, Epochs).
- **`runner.py`**: The main entry point for all diagnostics.

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Diagnostics (Runner)
To run the automated benchmarking tool:
```bash
python runner.py
```
*The system will interactively ask if you want to use a custom dataset or the built-in clinical benchmark.*

## 🔹 Output Format
The system provides:
*   **Performance Metrics Table**: Side-by-side comparison of Classical vs. Quantum accuracy and execution time.
*   **Visual Reports**: `comparison_report.png` showing diagnostic trends.
*   **Final Conclusion**: A data-driven analysis of which model performs better for the specific clinical task.

---
*Built for High-Precision Hybrid Medical Analytics.*
