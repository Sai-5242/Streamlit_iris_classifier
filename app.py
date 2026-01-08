import streamlit as st
import joblib
import pandas as pd

st.title("🌸 IRIS Classification APP")

st.write(
    "The Iris Classification App predicts the species of iris flowers "
    "based on sepal length, sepal width, petal length, and petal width."
)

# Load model
model = joblib.load('models\model.pkl')

# Sidebar inputs
st.sidebar.header("Input Features")

sepal_length = st.sidebar.number_input(
    "Sepal Length", min_value=4.0, max_value=10.0, value=5.0, step=0.1
)
sepal_width = st.sidebar.number_input(
    "Sepal Width", min_value=2.0, max_value=6.0, value=3.0, step=0.1
)
petal_length = st.sidebar.number_input(
    "Petal Length", min_value=1.0, max_value=7.0, value=4.0, step=0.1
)
petal_width = st.sidebar.number_input(
    "Petal Width", min_value=0.1, max_value=2.5, value=1.0, step=0.1
)

# Input DataFrame
input_df = pd.DataFrame({
    "SepalLengthCm":[sepal_length] ,
    "SepalWidthCm": [sepal_width],
    "PetalLengthCm": [petal_length],
    "PetalWidthCm": [petal_width],
})

st.subheader("📊 Input Values")
st.write(input_df)

# Prediction
if st.button("🔮 Predict"):
    prediction = model.predict(input_df)
    st.success(f"The predicted species is: **{prediction[0]}**")
else:
    st.info("Click the button to predict the species of an iris flower.")