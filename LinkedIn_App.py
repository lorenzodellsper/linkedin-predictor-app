import streamlit as st
import pandas as pd
import joblib

# Load Model
log_reg = joblib.load("linkedin_logreg.joblib")

# Model feature order (must match training!)
FEATURES = ['income', 'education', 'parent', 'married', 'female', 'age']

# Option labels

INCOME_OPTIONS = {
    "1: Less than $10,000": 1,
    "2: $10k–$20k": 2,
    "3: $20k–$30k": 3,
    "4: $30k–$40k": 4,
    "5: $40k–$50k": 5,
    "6: $50k–$75k": 6,
    "7: $75k–$100k": 7,
    "8: $100k–$150k": 8,
    "9: $150k or more": 9,}

EDUCATION_OPTIONS = {
    "1: Less than high school": 1,
    "2: High school incomplete": 2,
    "3: High school graduate / GED": 3,
    "4: Some college, no degree": 4,
    "5: Two-year associate degree": 5,
    "6: Four-year Bachelor’s degree": 6,
    "7: Some postgraduate / professional": 7,
    "8: Postgraduate / professional degree": 8,}

# App Interface

st.title("LinkedIn Usage Predictor")

st.write(
    "This app uses a logistic regression model trained on survey data "
    "to predict whether someone is likely to use LinkedIn.")

st.header("Enter Individual Information")

# Income
income_label = st.selectbox(
    "Household Income",
    options=list(INCOME_OPTIONS.keys()),
    index=7)
income = INCOME_OPTIONS[income_label]

# Education
education_label = st.selectbox(
    "Education Level",
    options=list(EDUCATION_OPTIONS.keys()),
    index=6)
education = EDUCATION_OPTIONS[education_label]

# Parent (0/1)
parent_label = st.radio("Parent of a child under 18?", ["No", "Yes"])
parent = 1 if parent_label == "Yes" else 0

# Married (0/1)
married_label = st.radio("Married?", ["No", "Yes"])
married = 1 if married_label == "Yes" else 0

# Female (0/1)
female_label = st.radio("Gender", ["Not female", "Female"])
female = 1 if female_label == "Female" else 0

# Age
age = st.number_input("Age (years)", min_value=18, max_value=100, value=42, step=1)

# Build input row for model
input_df = pd.DataFrame(
    [[income, education, parent, married, female, age]],
    columns=FEATURES)

# Prediction
st.subheader("Prediction")

if st.button("Predict LinkedIn Usage"):
    prob = log_reg.predict_proba(input_df)[0][1]
    pred = log_reg.predict(input_df)[0]

    st.write(f"**Predicted probability of using LinkedIn:** {prob:.2%}")

    if pred == 1:
        st.success("The model predicts this person **uses LinkedIn**.")
    else:
        st.warning("The model predicts this person **does not use LinkedIn**.")

    st.caption(
        "Prediction is based on logistic regression analysis of survey data.")