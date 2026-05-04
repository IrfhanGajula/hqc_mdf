"""
HQC-MDF: Programmatic Integration Example
----------------------------------------
This script demonstrates how to use the Hybrid Quantum Framework inside another 
Python project without needing the command-line runner.
"""

# 1. Import the core pipeline manager
try:
    from hqc_mdf.hybrid_core.pipeline_manager import run_pipeline
except ImportError:
    print("Error: Could not import HQC-MDF. Ensure you are in the project folder or have run 'pip install .'")
    exit(1)

def run_custom_analysis():
    print("--- Starting Programmatic Diagnostic Exploration ---")

    # 2. Execute the hybrid pipeline
    # You can specify a custom medical CSV path here if you have one.
    # Setting train=True will train the quantum circuits for the best accuracy.
    results = run_pipeline(
        disease_name="breast_cancer", 
        train=True, 
        custom_path=None  # Replace with "path/to/your_data.csv" for own datasets
    )

    # 3. Access the underlying results programmatically
    md = results['metadata']
    print(f"\nAnalysis complete for: {md['name']}")
    print(f"Total Patient Samples Processed: {md['total_samples']}")
    print(f"Clinical Features Analyzed: {md['features']}")
    
    print("\n[SUCCESS] The diagnostic pipeline has finished!")
    print(f"-> Full audit data preserved in: diagnostic_results.json")
    print(f"-> Clinical visuals saved in: models/plots/")

if __name__ == "__main__":
    run_custom_analysis()
