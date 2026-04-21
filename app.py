import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("housing_model.joblib")

import json

with open("town_means.json", "r") as f:
    town_means = json.load(f)

with open("postcode_means.json", "r") as f:
    postcode_means = json.load(f)

global_mean = 12.5

st.title("UK House Price Predictor")
st.write("Enter the details of a property to get a price estimate.")

property_type = st.selectbox("Property type", ["Detached", "Semi-detached", "Terraced", "Flat"])
tenure = st.selectbox("Tenure", ["Freehold", "Leasehold"])
old_new = st.selectbox("Property age", ["Existing", "Newly built"])
town = st.text_input("Town or city", value="LONDON")
postcode = st.text_input("Postcode", value="SW1A 1AA")

property_type_D = 1 if property_type == "Detached" else 0
property_type_F = 1 if property_type == "Flat" else 0
property_type_S = 1 if property_type == "Semi-detached" else 0
property_type_T = 1 if property_type == "Terraced" else 0
property_type_O = 0

tenure_F = 1 if tenure == "Freehold" else 0
tenure_L = 1 if tenure == "Leasehold" else 0

old_new_N = 1 if old_new == "Existing" else 0
old_new_Y = 1 if old_new == "Newly built" else 0

town_encoded = town_means.get(town.upper(), global_mean)
postcode_encoded = postcode_means.get(postcode.upper(), global_mean)

import datetime

year = datetime.datetime.now().year
month = datetime.datetime.now().month

if st.button("Predict price"):
    input_data = pd.DataFrame([{
        "year": year,
        "month": month,
        "property_type_D": property_type_D,
        "property_type_F": property_type_F,
        "property_type_O": property_type_O,
        "property_type_S": property_type_S,
        "property_type_T": property_type_T,
        "tenure_F": tenure_F,
        "tenure_L": tenure_L,
        "old_new_N": old_new_N,
        "old_new_Y": old_new_Y,
        "town_encoded": town_encoded,
        "postcode_encoded": postcode_encoded
    }])

    log_prediction = model.predict(input_data)[0]
    price = np.exp(log_prediction)
    
    st.success(f"Estimated price: £{price:,.0f}")