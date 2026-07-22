import streamlit as st
import pandas as pd
import pickle

# Load saved transformer and model dictionary
with open("Diamond.pkl", "rb") as f:
    saved_objects = pickle.load(f)
    transformer = saved_objects['transformer']
    model = saved_objects['model']

st.title("💎 Diamond Price Prediction App")
st.write("Enter diamond details to predict price")

# ---------------- USER INPUTS ----------------
carat = st.number_input("Carat", min_value=0.1, step=0.01)

cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

depth = st.number_input("Depth", min_value=0.0)
table = st.number_input("Table", min_value=0.0)
x = st.number_input("X dimension", min_value=0.0)
y = st.number_input("Y dimension", min_value=0.0)
z = st.number_input("Z dimension", min_value=0.0)

# ---------------- RAW DATAFRAME ----------------
# Build DataFrame matching the exact original feature structure
input_data = pd.DataFrame([[
    carat, cut, color, clarity, depth, table, x, y, z
]], columns=["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"])

# ---------------- PREDICT ----------------
if st.button("Predict Price 💰"):
    # Apply fitted ColumnTransformer (encodes categoricals & scales numericals automatically)
    input_transformed = transformer.transform(input_data)
    
    # Predict price
    prediction = model.predict(input_transformed)
    st.success(f"Estimated Diamond Price: 💵 ${prediction[0]:.2f}")