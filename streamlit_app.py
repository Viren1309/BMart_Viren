from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'brightmart_model.joblib'

model = joblib.load(MODEL_PATH)

st.title('Bright Mart Advertising Sales Predictor')
st.write('Estimate sales from TV, Radio, and Newspaper spend.')

col1, col2, col3 = st.columns(3)
with col1:
    tv = st.number_input('TV spend (₹000)', min_value=0.0, value=150.0, step=5.0)
with col2:
    radio = st.number_input('Radio spend (₹000)', min_value=0.0, value=25.0, step=5.0)
with col3:
    newspaper = st.number_input('Newspaper spend (₹000)', min_value=0.0, value=20.0, step=5.0)

features = pd.DataFrame(
    [[tv, radio, newspaper]],
    columns=['TV', 'Radio', 'Newspaper']
)

prediction = model.predict(features)[0]

st.metric('Predicted Sales', f'₹{prediction:.2f} million')

st.caption('Model: Multiple Linear Regression')
st.caption('Formula: Sales = Intercept + TV*coef + Radio*coef + Newspaper*coef')
