import glob
import os
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Starbucks Category Predictor", page_icon="☕")
st.title("☕ Starbucks Beverage Category Predictor")
st.write("Enter the nutrition values of a drink and the model will predict its category.")

# Find the saved model file in the same folder as this app
HERE = os.path.dirname(os.path.abspath(__file__))
matches = sorted(glob.glob(os.path.join(HERE, "starbucks_decision_tree_model*.pkl")))
if not matches:
    st.error("Model file (.pkl) not found. Upload it to the same GitHub folder as app.py.")
    st.stop()


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


artifacts = load_model(matches[0])
model = artifacts["model"]
le = artifacts["label_encoder"]
features = artifacts["feature_names"]

values = {}
col1, col2 = st.columns(2)
for i, name in enumerate(features):
    with (col1 if i % 2 == 0 else col2):
        values[name] = st.number_input(name, min_value=0.0, value=0.0, step=1.0)

if st.button("Predict category", type="primary"):
    X = pd.DataFrame([values], columns=features)
    pred = model.predict(X)
    st.success(f"Predicted category: **{le.inverse_transform(pred)[0]}**")
