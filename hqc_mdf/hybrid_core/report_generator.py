from fpdf import FPDF
import datetime
import os
import json
import subprocess
import platform

class DiagnosticReport(FPDF):
    def header(self):
        # Logo placeholder (optional)
        # self.image('logo.png', 10, 8, 33)
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, 'HQC-MDF: HYBRID QUANTUM CLINICAL REPORT', border=0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_font('helvetica', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', border=0, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Page {self.page_no()} | Confidential Medical Diagnostic Document', border=0, align='C')

def generate_pdf_report(prediction_data, output_path="Medical_Diagnosis_Report.pdf"):
    """
    Generates a polished PDF report combining benchmarks, visuals, and prediction results.
    """
    pdf = DiagnosticReport()
    pdf.add_page()
    
    # 1. Patient & System Metadata
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '1. DATASET & SYSTEM PROFILE', border=0, new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.set_font('helvetica', '', 10)
    
    # Load metadata from json if available
    try:
        with open('diagnostic_results.json', 'r') as f:
            manifest = json.load(f)
            meta = manifest.get('metadata', {})
            metrics = manifest.get('metrics', {})
            conclusion_text = manifest.get('conclusion', 'No conclusion found.')
    except Exception:
        meta = {}
        metrics = {}
        conclusion_text = "Analysis pending."

    pdf.cell(50, 8, f"Source Dataset:", border=0)
    pdf.cell(0, 8, f"{meta.get('name', 'Custom Dataset')}", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 8, f"Samples Analyzed:", border=0)
    pdf.cell(0, 8, f"{meta.get('total_samples', 'N/A')}", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(50, 8, f"Feature Dimensions:", border=0)
    pdf.cell(0, 8, f"{meta.get('features', 'N/A')} PCA-reduced to Quantum State", border=0, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # 2. Hybrid Performance Benchmarks
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '2. HYBRID PERFORMANCE BENCHMARKS', border=0, new_x="LMARGIN", new_y="NEXT", align='L')
    
    # Table Header
    pdf.set_font('helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(45, 8, 'Model', border=1, align='C', fill=True)
    pdf.cell(30, 8, 'Accuracy', border=1, align='C', fill=True)
    pdf.cell(30, 8, 'Precision', border=1, align='C', fill=True)
    pdf.cell(30, 8, 'Recall', border=1, align='C', fill=True)
    pdf.cell(30, 8, 'Time (s)', border=1, new_x="LMARGIN", new_y="NEXT", align='C', fill=True)
    
    pdf.set_font('helvetica', '', 9)
    for model_name, m in metrics.items():
        pdf.cell(45, 8, str(model_name), border=1, align='L')
        pdf.cell(30, 8, f"{m.get('Accuracy', 0):.4f}", border=1, align='C')
        pdf.cell(30, 8, f"{m.get('Precision', 0):.4f}", border=1, align='C')
        pdf.cell(30, 8, f"{m.get('Recall', 0):.4f}", border=1, align='C')
        pdf.cell(30, 8, f"{m.get('Time', 0):.4f}", border=1, new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)

    # 3. Visual Analytics (Images)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '3. VISUAL ANALYTICS', border=0, new_x="LMARGIN", new_y="NEXT", align='L')
    
    plot_dir = "models/plots"
    if os.path.exists(plot_dir):
        # We'll put two plots per row or just stack them
        plots = ["accuracy_benchmarks.png", "clinical_reliability.png"]
        y_pos = pdf.get_y()
        for i, plot in enumerate(plots):
            plot_path = os.path.join(plot_dir, plot)
            if os.path.exists(plot_path):
                # Image(name, x, y, w, h)
                pdf.image(plot_path, x=10 + (i*95), y=y_pos, w=90)
        pdf.ln(70) # Move down after images

    # 4. Final Clinical Prediction (The most important part)
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '4. PATIENT-SPECIFIC DIAGNOSTIC CONSENSUS', border=0, new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.ln(5)
    
    # Prediction Box
    pdf.set_fill_color(245, 247, 250)
    pdf.set_draw_color(0, 102, 204) # Medical Blue
    pdf.set_line_width(0.5)
    
    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 10, f" Patient Status: {prediction_data['status']}", border=1, new_x="LMARGIN", new_y="NEXT", align='L', fill=True)
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 8, f" Overall Confidence: {prediction_data['confidence']:.1f}%\n Consensus Basis: {prediction_data['consensus']}", border=1, align='L', fill=True)
    pdf.ln(10)
    
    # 5. Clinical Summary Text
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, '5. CLINICAL INTERPRETATION & CONCLUSION', border=0, new_x="LMARGIN", new_y="NEXT", align='L')
    pdf.set_font('helvetica', '', 11)
    pdf.multi_cell(0, 7, prediction_data['summary'])
    
    pdf.ln(10)
    pdf.set_font('helvetica', 'I', 10)
    pdf.multi_cell(0, 6, "Disclaimer: This report is generated by an experimental Hybrid Quantum Framework. It should be used as a decision-support tool only and must be verified by a licensed medical professional.")

    # Save
    pdf.output(output_path)
    print(f"\n[REPORT] Formal PDF report generated successfully: {output_path}")
    return output_path

def open_file(filepath):
    """
    Opens a file using the system's default application.
    """
    try:
        if platform.system() == 'Windows':
            os.startfile(filepath)
        elif platform.system() == 'Darwin': # macOS
            subprocess.call(['open', filepath])
        else: # Linux
            subprocess.call(['xdg-open', filepath])
        return True
    except Exception as e:
        print(f"[ERROR] Could not open file {filepath}: {e}")
        return False
