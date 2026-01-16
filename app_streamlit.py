import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF
from inference.plausibility import check_input_plausibility, soft_consistency_check
from inference.adaptive_logic import perform_adaptive_inference
from inference.confidence import generate_final_output

# --- PDF GENERATOR FUNCTION ---
def create_pdf_report(patient_name, case_id, user_input, result):
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Adaptive Anemia Classification Report", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(200, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)

    # Personalization Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 7, f"Patient Name: {patient_name if patient_name else 'N/A'}", ln=True)
    pdf.cell(0, 7, f"Case ID: {case_id if case_id else 'N/A'}", ln=True)
    pdf.ln(5)

    # Input Data Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. CBC Parameters (Input)", ln=True)
    pdf.set_font("Arial", '', 10)
    # Grouping parameters for better readability in PDF
    for key, val in user_input.items():
        pdf.cell(0, 7, f"- {key}: {val}", ln=True)
    pdf.ln(5)

    # Diagnosis Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. System Diagnosis", ln=True)
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(0, 51, 102) # Medical Blue
    pdf.cell(0, 10, f"Predicted Condition: {result['diagnosis']}", ln=True)
    pdf.set_text_color(0, 0, 0) # Reset to Black
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 7, f"Confidence Level: {result['confidence']['tier']}", ln=True)
    pdf.cell(0, 7, f"Reasoning: {result['confidence']['logic_flags']}", ln=True)
    pdf.ln(5)

    # Probabilities
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Differential Diagnosis Probabilities", ln=True)
    pdf.set_font("Arial", '', 10)
    for diag, prob in result['probabilities'].items():
        if prob > 0.01: # Only show relevant probabilities > 1%
            pdf.cell(0, 7, f"- {diag}: {prob * 100:.2f}%", ln=True)
    pdf.ln(10)

    # Warnings Section
    if result['warnings']:
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(200, 0, 0) # Red
        pdf.cell(0, 10, "4. Physiological Warnings", ln=True)
        pdf.set_font("Arial", '', 10)
        for w in result['warnings']:
            pdf.cell(0, 7, f"!! {w}", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

    # Final Disclaimer
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, f"LEGAL DISCLAIMER: {result['disclaimer']}")
    
    return pdf.output(dest='S').encode('latin-1')

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Physiology-Aware Anemia Classifier", layout="wide")

st.title("🩸 Adaptive Anemia Classification System")
st.markdown("---")

# --- SIDEBAR: PERSONALIZATION & INPUT ---
st.sidebar.header("📋 Case Information")
patient_name = st.sidebar.text_input("Patient Name", placeholder="Enter Name")
case_id = st.sidebar.text_input("Case / ID Number", placeholder="e.g. CBC-102")

st.sidebar.markdown("---")
st.sidebar.header("🧪 User CBC Input")

# Manual inputs starting at 0.0
hgb = st.sidebar.number_input("Hemoglobin (HGB)", 0.0, 25.0, 0.0, 0.1)
rbc = st.sidebar.number_input("Red Blood Cell (RBC)", 0.0, 10.0, 0.0, 0.1)
mcv = st.sidebar.number_input("Mean Corpuscular Vol (MCV)", 0.0, 150.0, 0.0, 1.0)
hct = st.sidebar.number_input("Hematocrit (HCT)", 0.0, 500.0, 0.0, 0.1)
wbc = st.sidebar.number_input("White Blood Cell (WBC)", 0.0, 100.0, 0.0, 0.1)
plt = st.sidebar.number_input("Platelets (PLT)", 0.0, 1000.0, 0.0, 1.0)
pdw = st.sidebar.number_input("PDW", 0.0, 25.0, 0.0, 0.1)

# Percentages for Inflammation Index
neut = st.sidebar.slider("Neutrophil % (NEUTp)", 0, 100, 0)
lym = st.sidebar.slider("Lymphocyte % (LYMp)", 0, 100, 0)

# Create the input dictionary
user_input = {
    "HGB": hgb, "RBC": rbc, "MCV": mcv, "HCT": hct,
    "WBC": wbc, "PLT": plt, "PDW": pdw,
    "NEUTp": neut, "LYMp": lym
}

# --- MAIN PAGE: LOGIC ---

# Check for input before running
if hgb == 0 and rbc == 0:
    st.info("👋 **System Ready.** Please enter CBC parameters in the sidebar to generate a diagnostic profile.")
    
    # 
    
else:
    # 1. PLAUSIBILITY GATE
    is_plausible, error_msg = check_input_plausibility(user_input)

    if not is_plausible:
        st.error(f"❌ **Plausibility Gate Failed:** {error_msg}")
        st.warning("Physiological conflict detected. Inference blocked for safety.")
    else:
        st.success("✅ **Plausibility Gate Passed.**")
        
        # 2. INFERENCE
        try:
            warnings = soft_consistency_check(user_input)
            final_probs, violation_detected = perform_adaptive_inference(user_input)
            result = generate_final_output(final_probs, violation_detected, warnings)

            # 3. DISPLAY RESULTS
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Final Diagnosis")
                st.metric(label="Predicted Type", value=result["diagnosis"])
                
                conf_tier = result["confidence"]["tier"]
                color = "green" if "High" in conf_tier else "orange" if "Moderate" in conf_tier else "red"
                st.markdown(f"Confidence Tier: **:{color}[{conf_tier}]**")
                
                st.info(f"**Reasoning:** {result['confidence']['logic_flags']}")
                
                # PDF BUTTON
                pdf_data = create_pdf_report(patient_name, case_id, user_input, result)
                st.download_button(
                    label="📄 Download Diagnostic PDF",
                    data=pdf_data,
                    file_name=f"{patient_name.replace(' ', '_')}_Anemia_Report.pdf" if patient_name else "Anemia_Report.pdf",
                    mime="application/pdf"
                )

            with col2:
                st.subheader("Probability Distribution")
                prob_df = pd.DataFrame(list(result["probabilities"].items()), columns=["Anemia Type", "Probability"])
                st.bar_chart(prob_df.set_index("Anemia Type"))

            # 4. SYSTEM LOGIC INSIGHTS
            st.markdown("---")
            st.subheader("🧠 Adaptive Logic Insights")
            
            # 
            
            if violation_detected:
                st.warning("⚠️ **Path B Warning:** Physiological contradiction detected between indices.")
            else:
                st.success("Path B: Data is consistent with known medical rules.")
            
            if warnings:
                st.write("**Observations:**")
                for w in warnings:
                    st.write(f"🔍 *{w}*")

        except Exception as e:
            st.error(f"Inference Error: {e}")