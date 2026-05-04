import os
import sys

# Ensure the project root is in the path
sys.path.append(os.getcwd())

from hqc_mdf.hybrid_core.pipeline_manager import run_pipeline

def main():
    print("\n" + "="*60)
    print("   [ HQC-MDF ] HYBRID QUANTUM MEDICAL DIAGNOSTICS SYSTEM")
    print("   Advanced Clinical Benchmarking & Analytics")
    print("="*60 + "\n")
    
    print("WELCOME! This runner will guide you through the diagnostic process.")
    print("You can use a custom medical CSV or run the automated benchmark.")
    
    # 1. Dataset Selection
    print("\n[STEP 1/2] Dataset Configuration")
    custom_path = input("Enter the path to your medical CSV (or press Enter for default Breast Cancer): ").strip()
    
    import pandas as pd
    
    disease = "breast_cancer"
    path_to_use = None
    
    if custom_path:
        # Check if user provided quotes and strip them
        custom_path = custom_path.strip('\"\'')
        if os.path.exists(custom_path):
            print(f"\n[OK] Found clinical dataset at: {custom_path}")
            path_to_use = custom_path
            disease = "Custom Dataset"
            
            # PREVIEW DATA FOR TRANSPARENCY
            print("\n[DATA SNAPSHOT - Previewing Source File]")
            try:
                preview_df = pd.read_csv(custom_path)
                print(f"Columns Detected: {', '.join(preview_df.columns[:5])} ...")
                print(preview_df.head(5))
                print(f"Total Rows in File: {len(preview_df)}")
            except Exception as e:
                print(f"ERROR reading file: {e}")
                sys.exit(1)
        else:
            print(f"\n[FATAL ERROR] File not found at '{custom_path}'")
            print("To ensure data integrity, the system will not fall back to defaults.")
            print("Please check the file path and run again.")
            sys.exit(1)
    else:
        print("\nUsing default Wisconsin Breast Cancer dataset for demonstration.")
        print("Total Records: 569 | Features: 30")

    # 2. Pipeline Execution Mode
    print("\n[STEP 2/2] Diagnostic Mode")
    retrain = input("Do you want to retrain the models on this data? (y/n) [default: n]: ").lower().strip() == 'y'
    
    # 3. Trigger Pipeline
    try:
        results = run_pipeline(
            disease_name=disease, 
            train=retrain, 
            custom_path=path_to_use
        )
        
        print("\n" + "-"*60)
        print("SUCCESS: All diagnostic processes completed!")
        print(f"Reports saved: models/plots/ (Accuracy, Reliability, Overhead)")
        print(f"Models saved: models/ directory")
        print("-" * 60 + "\n")
        
        # 4. Immediate Prediction Trigger
        run_pred = input("Would you like to run a live prediction on these trained models? (y/n) [default: y]: ").lower().strip()
        prediction_results = None
        if run_pred != 'n':
            from predict import run_prediction
            prediction_results = run_prediction(disease, custom_path=path_to_use)

        # 5. PDF Report Generation Layer
        if prediction_results:
            print("\n" + "="*60)
            gen_pdf = input("Do you want to download the final diagnostic report with graphs as a PDF? (y/n) [default: y]: ").lower().strip()
            if gen_pdf != 'n':
                print(f"\n[REPORT] Formal PDF report generated successfully: Medical_Diagnosis_Report.pdf")
                from hqc_mdf.hybrid_core.report_generator import generate_pdf_report, open_file
                report_path = generate_pdf_report(prediction_results)
                
                open_now = input("Would you like to open the report now? (y/n) [default: y]: ").lower().strip()
                if open_now != 'n':
                    open_file(report_path)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR during pipeline execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
