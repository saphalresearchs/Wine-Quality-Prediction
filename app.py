import streamlit as st
import joblib
import numpy as np

model = joblib.load("wine_prediction_model.pkl")

st.title("Wine Quality predictor")
st.write("Write given features of the wine.")

fixed_acidity = st.number_input("Fixed Acidity")
Volatile_acidity = st.number_input("Volatile Acidity" )
Citric_acid = st.number_input("Citric Acid" )
Residual_sugar = st.number_input("Residual Sugar" )
Chlorides = st.number_input("Chlorides", step=0.001, format="%.4f")
Free_Sulphur_Dioxide = st.number_input("Free Sulphur Dioxide" )
Total_sulphur_dioxide = st.number_input("Total Sulphur Dioxide" )
Density = st.number_input("Density", step=0.0001, format="%.4f")
pH = st.number_input("pH" )
sulphates = st.number_input("Sulphates" )
Alcohol = st.number_input("Alcohol" )


if st.button("Check Quality"):
    input_array = [fixed_acidity, Volatile_acidity, Citric_acid, Residual_sugar, Chlorides, Free_Sulphur_Dioxide, Total_sulphur_dioxide, Density, pH, sulphates, Alcohol]
    input_array_asarray = np.asarray(input_array)
    input_reshaped = input_array_asarray.reshape(1,-1)
    prediction = model.predict(input_reshaped)

    if prediction[0]==1:
        st.success("Good Wine Quality")
    else:
        st.error("Bad Wine Quality")