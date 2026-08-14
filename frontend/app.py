import streamlit as st
import requests
import pandas as pd
import os

API_URL = "https://fraud-detector-backend-kzuo.onrender.com/predict"

# @st.cache_data means run this function once, then reuse the result on
# every rerun instead of reloading the whole CSV from disk every single time someone clicks a button
@st.cache_data
def load_data():
    # Build the path relative to this script's own location, not
    # whatever folder happened to be "current" when it was launched --
    # this makes it work the same locally and on Streamlit Cloud.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "training", "data", "test_set.csv")
    return pd.read_csv(csv_path)

data = load_data()

st.title("Fraud Detector")
st.write("Load a random real transaction or edit the values, then check it.")

if "transaction" not in st.session_state:  # session_state lets Streamlit remember information while the user interacts
    st.session_state.transaction = data[data["Class"] == 0].iloc[0].drop("Class").to_dict()

col1, col2 = st.columns(2)
with col1:
    if st.button("Load random fraud example"): #only keeps rows where class=1 (fraud)
        random_fraud_row = data[data["Class"] == 1].sample(1).iloc[0]
        st.session_state.transaction = random_fraud_row.drop("Class").to_dict()
with col2:
    if st.button("Load random legitimate example"):  #only keeps rows where class=2 (legitamate)
        random_legit_row = data[data["Class"] == 0].sample(1).iloc[0]
        st.session_state.transaction = random_legit_row.drop("Class").to_dict()

inputs = {} #stores the request data in an empty python dictionary
inputs["Time"] = st.number_input("Time", value=float(st.session_state.transaction["Time"]))
inputs["Amount"] = st.number_input("Amount", value=float(st.session_state.transaction["Amount"]))

with st.expander("Raw features (V1–V28) — auto-filled from example"):
    for i in range(1, 29):
        key = f"V{i}"
        inputs[key] = st.number_input(key, value=float(st.session_state.transaction[key]), format="%.6f")

if st.button("Check transaction", type="primary"): #this section sends the request and prints the result 
    try:   # "try" means Python will try to run this code.  will handle the error instead of crashing the app.
        response = requests.post(API_URL, json=inputs) #requests.post sends whatever's currently in the form to the backend's /predict endpoint
        response.raise_for_status()
        result = response.json()

        if result["prediction"] == "fraud":
            st.error(f"🚨 FRAUD — confidence {result['confidence']:.2%}")
        else:
            st.success(f"✅ Legitimate — confidence {result['confidence']:.2%}")
    except requests.exceptions.ConnectionError:
        st.warning("Can't reach the backend. Is it running? (`uvicorn app.main:app --reload` in `backend/`)")