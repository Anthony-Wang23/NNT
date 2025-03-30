import pandas as pd
import pickle
import joblib
import streamlit as st
import os  # 替代 pathlib

# 模型路径配置
MODEL_PATH = 'best_mlp_model.pkl'

# 验证文件存在
if not os.path.exists(MODEL_PATH):
    st.error(f"模型文件未找到: {os.path.abspath(MODEL_PATH)}")
    st.stop()

@st.cache_resource
def load_model():
    try:
        # 尝试 joblib
        try:
            model = joblib.load(MODEL_PATH)
            if hasattr(model, 'predict_proba'):
                return model
        except:
            pass
        
        # 尝试 pickle
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            if hasattr(model, 'predict_proba'):
                return model
        
        raise Exception("无效的模型文件")
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        st.stop()

# 加载模型
model = load_model()

# 其余原有代码保持不变...

# Page configuration
st.set_page_config(
    page_title="Melanoma SLN Metastasis Predictor",
    layout="centered",
    page_icon=":hospital:"
)

# Custom styling
st.markdown("""
<style>
    .header-style { font-size:24px; font-weight:bold; color:#2a52be; }
    .result-positive { color:#d62728; font-weight:bold; font-size:28px; }
    .result-negative { color:#2ca02c; font-weight:bold; font-size:28px; }
    .feature-header { font-size:18px; font-weight:bold; color:#2a52be; margin-top:15px; }
</style>
""", unsafe_allow_html=True)



# App header
st.title("Melanoma Sentinel Lymph Node Metastasis Predictor")
st.markdown("<div class='header-style'>Clinical Decision Support Tool</div>", unsafe_allow_html=True)

# Input features
with st.form("patient_parameters"):
    st.markdown("<div class='feature-header'>Tumor Characteristics</div>", unsafe_allow_html=True)
    breslow = st.slider("Breslow Thickness (mm)", 0.0, 10.0, 4.0, 0.1,
                        help="Measured depth of tumor invasion")
    ki67 = st.slider("Ki67 Proliferation Index (%)", 0.0, 100.0, 0.0, 0.1,
                     help="Percentage of Ki67-positive tumor cells")

    st.markdown("<div class='feature-header'>Clinical Features</div>", unsafe_allow_html=True)
    subungual = st.radio("Subungual Melanoma?", options=["No", "Yes"],
                         help="Is the melanoma located under the nail?")
    treatment = st.radio("Prior Treatment Received?", options=["No", "Yes"],
                         help="Has the patient received any prior treatment?")

    submitted = st.form_submit_button("Calculate Metastasis Risk")

# Prepare input data
if submitted:
    input_data = pd.DataFrame({
        'Subtype': [1 if subungual == "Yes" else 0],
        'Breslow_Thickness': [breslow],
        'Ki67': [ki67],
        'Supplementary_Check': [1 if treatment == "Yes" else 0]
    })

    # Ensure consistent column order
    expected_columns = ['Subtype', 'Breslow_Thickness', 'Ki67', 'Supplementary_Check']
    input_data = input_data[expected_columns]

    with st.spinner("Analyzing patient data..."):
        try:
            # Get prediction probabilities
            predicted_probs = model.predict_proba(input_data)
            probability = predicted_probs[0][1]  # Get positive class probability

            # Display prediction result
            st.markdown("---")
            if probability >= 0.5:
                st.markdown(
                    f"<div class='result-positive'>High Risk: {probability:.1%} probability of metastasis</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-negative'>Low Risk: {probability:.1%} probability of metastasis</div>",
                            unsafe_allow_html=True)

            # Display input values for reference
            st.markdown("**Patient Parameters Used:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"- Breslow Thickness: {breslow} mm")
                st.write(f"- Ki67 Index: {ki67}%")
            with col2:
                st.write(f"- Subungual Melanoma: {subungual}")
                st.write(f"- Prior Treatment: {treatment}")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check your input values and try again")

# Footer
st.markdown("---")
st.markdown("""
*Clinical Decision Support Tool v1.0*  
*For physician use only - Not a substitute for clinical judgment*
""")
        #streamlit run D:/anaconda3/envs/py312/streamlit.py
