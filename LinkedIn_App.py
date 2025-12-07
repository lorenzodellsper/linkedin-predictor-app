import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import date

# Load trained model
log_reg = joblib.load("linkedin_logreg.joblib")

FEATURES = ['income', 'education', 'parent', 'married', 'female', 'age']

# Option Label Dropdown
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

# Sidebar about me
st.sidebar.title("About this app")
st.sidebar.write("**Author:** Lorenzo Della Speranza")
st.sidebar.write("**Program:** Georgetown MSBA")
st.sidebar.write(f"**Date:** {date.today().strftime('%B %d, %Y')}")
st.sidebar.write("**Model:** Logistic Regression")

# Main Title
st.title("LinkedIn Usage Predictor")

st.write(
    "This app uses a logistic regression model trained on a Pew Research "
    "Center survey to predict whether someone is likely to use LinkedIn.")

# Tabs
tab_pred, tab_about, tab_det = st.tabs(
    ["Prediction", "About the Data", "Model Details"])

# Prediction Tab
with tab_pred:
    st.header("Enter Individual Information")

    income_label = st.selectbox(
        "Household Income",
        options=list(INCOME_OPTIONS.keys()),
        index=7)
    income = INCOME_OPTIONS[income_label]

    education_label = st.selectbox(
        "Education Level",
        options=list(EDUCATION_OPTIONS.keys()),
        index=6)
    education = EDUCATION_OPTIONS[education_label]

    parent_label = st.radio("Parent of a child under 18?", ["No", "Yes"])
    parent = 1 if parent_label == "Yes" else 0

    married_label = st.radio("Married?", ["No", "Yes"])
    married = 1 if married_label == "Yes" else 0

    female_label = st.radio("Gender", ["Not female", "Female"])
    female = 1 if female_label == "Female" else 0

    age = st.number_input("Age (years)", min_value=18, max_value=100, value=42, step=1)

    # Build input row
    input_df = pd.DataFrame(
        [[income, education, parent, married, female, age]],
        columns=FEATURES)

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

# About the Data Tab
with tab_about:
    st.header("About the Data")
    st.markdown(
        """
- **Source:** Pew. 
- **Data for educational purposes only.**  
- **Target in this app:** Whether a respondent uses LinkedIn  
- **Predictors for this app:**
  - Household income (coded 1–9)
  - Education level (coded 1–8)
  - Parent of child under 18 (0/1)
  - Married (0/1)
  - Female (0/1)
  - Age (years)
- **Sample size after cleaning:** 1,260 respondents  
- Values such as *“don’t know”* or out-of-range codes were removed.
        """)

# Model Performance Tab
with tab_det:
    st.header("Model Explanation")

    st.markdown(
        """
This model is a logistic regression, so each predictor has a **coefficient**.  
Positive coefficients increase the predicted probability of using LinkedIn;  
negative coefficients decrease it (holding other variables constant).
        """)

    coefs = log_reg.coef_[0]
    coef_df = pd.DataFrame(
        {"Feature": FEATURES, "Coefficient": coefs}
    ).sort_values("Coefficient", ascending=False)

    st.subheader("Coefficient Plot")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(coef_df["Feature"], coef_df["Coefficient"])
    ax.set_xlabel("Coefficient value")
    ax.set_ylabel("Predictor")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.caption(
        "Larger positive coefficients indicate predictors associated with a higher likelihood "
        "of LinkedIn use in this survey sample.")

    st.header("Model Performance")
    
    # Hard-coded from model results
    accuracy = 0.67
    precision_1 = 0.50
    recall_1 = 0.74
    f1_1 = 0.60
    
    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write(f"**Precision:** {precision_1:.2f}")
    st.write(f"**Recall:** {recall_1:.2f}")
    st.write(f"**F1 score:** {f1_1:.2f}")
    
    st.caption(
        "Metrics are based on a separate 20% test set that was not used to train the model." )
