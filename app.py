import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline

# Page title
st.title("📉 Customer Churn Prediction App")

# Load the trained model
model = None
try:
    with open("churn_prediction_model.pkl", "rb") as file:
        model = pickle.load(file)

    if isinstance(model, Pipeline):
        st.success("✅ Model loaded successfully!")
    else:
        st.error(f"❌ Loaded object is not a valid model! It is of type: {type(model)}")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

if model is not None:
    # Sidebar for input data
    st.sidebar.header("🧾 Input Customer Data")

    credit_score = st.sidebar.number_input("Credit Score", min_value=0)
    country = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
    age = st.sidebar.number_input("Age", min_value=18, max_value=100)
    balance = st.sidebar.number_input("Account Balance", min_value=0.0)
    products_number = st.sidebar.number_input("Number of Products", min_value=1)
    credit_card = st.sidebar.selectbox("Credit Card", [0, 1])
    active_member = st.sidebar.selectbox("Active Member", [0, 1])
    estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    tenure = st.sidebar.number_input("Tenure (months)", min_value=0)

    # Prepare input dataframe
    input_data = pd.DataFrame({
        'credit_score': [credit_score],
        'gender': [gender],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary],
    })

    # Encode gender
    input_data['gender'] = input_data['gender'].map({"Male": 1, "Female": 0})

    # Manually one-hot encode country
    country_dict = {
        'country_France': 0,
        'country_Germany': 0,
        'country_Spain': 0
    }
    selected_country_col = f'country_{country}'
    country_dict[selected_country_col] = 1

    for col in country_dict:
        input_data[col] = country_dict[col]

    # Prediction
    if st.button("🚀 Predict Churn"):
        try:
            prediction = model.predict(input_data)
            pred_proba = model.predict_proba(input_data)[0][1]

            if prediction[0] == 1:
                st.error(f"❌ Customer is likely to churn. Probability: {pred_proba:.2f}")
            else:
                st.success(f"✅ Customer is not likely to churn. Probability: {pred_proba:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
else:
    st.warning("📂 Please upload a valid model to proceed.")
