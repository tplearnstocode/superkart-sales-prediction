import io
import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend (container name on the Docker network)
BACKEND_URL = "http://backend:7860"

# Same feature engineering as used during training
PRODUCT_ID_PREFIXES = {"FD": "FD - Food", "DR": "DR - Drinks", "NC": "NC - Non-Consumable"}
# Valid Product_Type_Category values for each prefix (as seen in the training data):
# only food (FD) can be perishable; drinks (DR) and non-consumables (NC) are always non perishable
PREFIX_CATEGORIES = {
    "FD": ["Perishables", "Non Perishables"],
    "DR": ["Non Perishables"],
    "NC": ["Non Perishables"]
}
REFERENCE_YEAR = 2026  # Store_Age_Years = 2026 - Store_Establishment_Year (as in the preprocessing)

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")
st.write("Forecast the total sales revenue of a product in a SuperKart store.")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product features
product_id_char = st.selectbox("Product Id Prefix", list(PRODUCT_ID_PREFIXES),
                               format_func=lambda prefix: PRODUCT_ID_PREFIXES[prefix])

# The category options depend on the selected prefix, so the two can never contradict each other
product_type_category = st.selectbox("Product Type Category", PREFIX_CATEGORIES[product_id_char])

product_weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, step=0.1, value=12.66)

# Non-consumables are always "No Sugar"; food and drinks are "Low Sugar" or "Regular"
sugar_options = ["No Sugar"] if product_id_char == "NC" else ["Low Sugar", "Regular"]
product_sugar_content = st.selectbox("Product Sugar Content", sugar_options)
product_allocated_area = st.number_input("Product Allocated Area (ratio of display area)", min_value=0.0, max_value=1.0,
                                         step=0.001, value=0.027, format="%.3f")
product_mrp = st.number_input("Product MRP", min_value=0.0, max_value=500.0, step=1.0, value=117.08)

# Collect user input for store features
store_establishment_year = st.number_input("Store Establishment Year", min_value=1950, max_value=REFERENCE_YEAR, step=1, value=2009)

# Calculated field: store age, computed exactly as in the data preprocessing step
store_age_years = REFERENCE_YEAR - store_establishment_year
st.text_input("Store Age (years) - calculated", value=str(store_age_years), disabled=True)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type1", "Supermarket Type2", "Food Mart"], index=2)

# Convert user input into a DataFrame (with the engineered features used by the model)
input_data = pd.DataFrame([{
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_MRP': product_mrp,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Product_Id_char': product_id_char,
    'Store_Age_Years': store_age_years,
    'Product_Type_Category': product_type_category
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=input_data.to_dict(orient='records')[0])  # Send data to Flask API
        if response.status_code == 200:
            prediction = response.json()['Predicted_Sales']
            st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
        else:
            st.error(f"Prediction failed: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")
st.write("Upload a CSV file with the columns: " + ", ".join(input_data.columns))

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        try:
            response = requests.post(f"{BACKEND_URL}/v1/predictbatch", files={"file": uploaded_file.getvalue()})  # Send file to Flask API
            if response.status_code == 200:
                predictions = response.json()
                batch_data = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                batch_data["Predicted_Sales"] = [predictions[str(i)] for i in batch_data.index]
                st.success("Batch predictions completed!")
                st.dataframe(batch_data)  # Display the input data along with the predictions
            else:
                st.error(f"Batch prediction failed: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Unable to connect to the prediction API.")
