import streamlit as st
import pandas as pd
import requests

# Page Config
st.set_page_config(page_title="RUL Predictor", layout="centered")

# Title with styling
st.markdown("""
    <h1 style='text-align: center; color: #2E86AB;'>🔧 RUL Prediction App</h1>
    <hr>
""", unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📎Please upload a CSV file to begin.",type=["csv"])

if uploaded_file is not None:
    try:
        user_input = pd.read_csv(uploaded_file, index_col=0)
        
        with st.expander("🔍 Preview Uploaded Data", expanded=False):
            st.dataframe(user_input, use_container_width=True)

        # Prediction button
        if st.button("🚀 Predict RUL", type='primary'):
            with st.spinner("Sending data to prediction server..."):
                url = "http://127.0.0.1:8000/predict"
                response = requests.post(url=url, json=user_input.to_dict(orient="records"))

                if response.status_code == 200:
                    result = response.json()
                    prediction = int(result['prediction'])

                    # st.markdown("### 🧠 Predicted RUL (Cycles):")
                    # st.markdown(f"{prediction}")
                    st.markdown(f"""
                        <div style='
                            background-color: #f1f8e9;
                            padding: 10px 15px;
                            border-radius: 8px;
                            border: 1px solid #dcedc8;
                            margin-top: 15px;
                            text-align: center;
                            width: 300px;
                            margin-left: auto;
                            margin-right: auto;
                        '>
                            <h4 style='color: #33691e; margin: 5px 0 10px;'>Predicted RUL:</h4>
                            <p style='font-size: 20px; font-weight: 600; color: #1b5e20; margin: 0;'>{prediction}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Prediction failed. Server responded with status code: {response.status_code}")

    except Exception as e:
        st.error(f"⚠️ Error loading CSV: `{e}`")

    

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with ⚙️ FastAPI + 🧊 Streamlit")
