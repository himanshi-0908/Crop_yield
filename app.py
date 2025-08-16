import streamlit as st
import numpy as np
import pickle
import sklearn

print(sklearn.__version__)

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Streamlit App
st.title("🌾 Crop Yield Prediction")

# Input fields (same as Flask form)
Year = st.text_input("Year")
average_rain_fall_mm_per_year = st.text_input("Average Rainfall (mm per year)")
pesticides_tonnes = st.text_input("Pesticides (tonnes)")
avg_temp = st.text_input("Average Temperature (°C)")
Area = st.text_input("Area")
Item = st.text_input("Crop Item")

# Prediction Button
if st.button("Predict"):
    try:
        features = np.array([[Year,
                              average_rain_fall_mm_per_year,
                              pesticides_tonnes,
                              avg_temp,
                              Area,
                              Item]], dtype=object)

        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1, -1)

        st.success(f"✅ Predicted Crop Yield: {prediction[0][0]:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
