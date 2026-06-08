import streamlit as st
import numpy as np
import pickle
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Chronic Kidney Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# ================= LOAD MODEL =================
model = pickle.load(open("ckd_model.pkl", "rb"))

# ================= PREMIUM MEDICAL THEME =================
st.markdown("""
<style>
.main {
    background-color: #f4f8fb;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #0b5394;
}
.stButton>button {
    background-color: #0077b6;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
.result-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.title("🩺 Chronic Kidney Disease Prediction")
st.markdown("### AI-powered Clinical Decision Support System")
st.markdown("---")

# ================= LAYOUT =================
col1, col2, col3 = st.columns(3)

# ========= COLUMN 1 =========
with col1:
    st.subheader("🧪 Basic Parameters")

    age = st.number_input("Age", 1, 120, 45)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.selectbox("Specific Gravity", [1.005,1.010,1.015,1.020,1.025])
    al = st.selectbox("Albumin", [0,1,2,3,4,5])
    su = st.selectbox("Sugar", [0,1,2,3,4,5])

    rbc = st.selectbox("Red Blood Cells", ["normal","abnormal"])
    pc = st.selectbox("Pus Cell", ["normal","abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["notpresent","present"])

# ========= COLUMN 2 =========
with col2:
    st.subheader("🧬 Blood Profile")

    ba = st.selectbox("Bacteria", ["notpresent","present"])
    bgr = st.number_input("Blood Glucose Random", 50, 500, 120)
    bu = st.number_input("Blood Urea", 1, 200, 40)
    sc = st.number_input("Serum Creatinine", 0.1, 20.0, 1.2)

    sod = st.number_input("Sodium", 100, 200, 135)
    pot = st.number_input("Potassium", 2.0, 10.0, 4.5)
    hemo = st.number_input("Hemoglobin", 3.0, 20.0, 13.5)
    pcv = st.number_input("Packed Cell Volume", 10, 60, 40)

# ========= COLUMN 3 =========
with col3:
    st.subheader("📋 Clinical Indicators")

    wc = st.number_input("White Blood Cell Count", 2000, 20000, 8000)
    rc = st.number_input("Red Blood Cell Count", 2.0, 8.0, 5.0)

    htn = st.selectbox("Hypertension", ["no","yes"])
    dm = st.selectbox("Diabetes Mellitus", ["no","yes"])
    cad = st.selectbox("Coronary Artery Disease", ["no","yes"])
    appet = st.selectbox("Appetite", ["good","poor"])
    pe = st.selectbox("Pedal Edema", ["no","yes"])
    ane = st.selectbox("Anemia", ["no","yes"])

# ================= ENCODER =================
def bin_map(val):
    return 1 if val in ["yes","present","abnormal","poor"] else 0

# ================= PREDICTION =================
st.markdown("---")

if st.button("🔍 Predict CKD Risk"):

    input_data = np.array([[
        age, bp, sg, al, su,
        bin_map(rbc), bin_map(pc), bin_map(pcc), bin_map(ba),
        bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
        bin_map(htn), bin_map(dm), bin_map(cad),
        bin_map(appet), bin_map(pe), bin_map(ane)
    ]])

    prediction = model.predict(input_data)[0]
    prob_ckd = model.predict_proba(input_data)[0][0]

    st.markdown("## 🧾 Clinical Result")

    if prediction == 0:
        result_text = "High Risk of CKD"
        st.error(f"⚠️ {result_text}\n\nRisk Probability: {prob_ckd:.2%}")
    else:
        result_text = "Low Risk of CKD"
        st.success(f"✅ {result_text}\n\nRisk Probability: {prob_ckd:.2%}")

    st.progress(float(prob_ckd))

    if prob_ckd > 0.75:
        st.warning("⚠️ Immediate medical consultation recommended.")
    elif prob_ckd > 0.40:
        st.info("🔎 Moderate risk — further tests advised.")
    else:
        st.success("✅ Low clinical risk.")

    # ================= DOWNLOAD REPORT =================
    report = f"""
CKD CLINICAL REPORT
-------------------------
Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}

Patient Age: {age}
Blood Pressure: {bp}

Prediction: {result_text}
CKD Probability: {prob_ckd:.2%}

Note: This is an AI-assisted prediction, not a medical diagnosis.
"""

    st.download_button(
        label="📥 Download Patient Report",
        data=report,
        file_name="CKD_Report.txt",
        mime="text/plain"
    )