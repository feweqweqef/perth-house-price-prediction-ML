#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# In[5]:


# Load the saved scaler and model
scaler = joblib.load('scaler.joblib')
model = joblib.load('best_lgb_model.pkl')
# Load the CSV file containing df3
df3 = pd.read_csv('df3.csv')


# In[6]:


import pandas as pd
import streamlit as st

# Create a function to get user inputs
def get_user_inputs(df_suburbs):
    st.header("Enter House Details")
    st.subheader("Basic Information")
    bedrooms = st.slider("🛏️ Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.slider("🚽 Number of Bathrooms", min_value=1, max_value=10, value=2)
    garage = st.slider("🚗 Number of Garages", min_value=0, max_value=5, value=2)
    land_area = st.number_input("🏞️ Land Area (in square meters)", min_value=0, value=300)
    floor_area = st.number_input("🏠 Floor Area (in square meters)", min_value=0, value=150)
    build_year = st.number_input("🏗️ Build Year", min_value=1800, value=2000)

    st.subheader("Distances (in meters)")
    cbd_dist = st.number_input("🏙️ Distance to CBD", min_value=0, value=1000)
    nearest_stn_dist = st.number_input("🚉 Distance to Nearest Station", min_value=0, value=500)
    nearest_sch_dist = st.number_input("🏫 Distance to Nearest School", min_value=0, value=2000)

    # Suburb selection using a dropdown list
    st.subheader("Suburb Status")
    selected_suburb = st.selectbox("🏢 Select a Suburb Status", ["Standard", "Luxury"])

    # Map selected suburb status to code (0 for "Standard", 1 for "Luxury")
    selected_suburb_code = 1 if selected_suburb == "Standard" else 0

    # Create a dictionary of user inputs
    user_inputs = {
        'BEDROOMS': bedrooms,
        'BATHROOMS': bathrooms,
        'GARAGE': garage,
        'LAND_AREA': land_area,
        'FLOOR_AREA': floor_area,
        'BUILD_YEAR': build_year,
        'CBD_DIST': cbd_dist,
        'NEAREST_STN_DIST': nearest_stn_dist,
        'NEAREST_SCH_DIST': nearest_sch_dist,
        'SUBURB_STATUS_CODE': selected_suburb_code
    }

    return user_inputs

# Create a function to make predictions
def predict_house_price(user_inputs):
    # Convert user inputs to a DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Scale the input features using the scaler
    input_scaled = scaler.transform(input_df)

    # Make predictions with the model
    predicted_price = model.predict(input_scaled)[0]
    return predicted_price

# Create the Streamlit app
def main():
    st.set_page_config(page_title="Perth House Price Prediction", layout="wide")
    
    # Background color and introduction
    st.markdown(
        """
        <style>
            body {
                background-color: #F0F0F0;
            }
            h1 {
                color: #003E7E;
            }
            h2 {
                color: #003E7E;
            }
            p {
                color: #555555;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Introduction
    st.title("Perth House Price Prediction")
    st.markdown("Welcome to the Perth House Price Prediction web app! With this interactive tool, you can predict the price of a house in Perth based on various features. Whether you are a homebuyer, a real estate investor, or simply curious about the housing market, this app is designed to help you make informed decisions. Enter the details of the house you are interested in, and let's get started!")
    
    logo_url = "https://img.freepik.com/premium-vector/house-real-estate-logo_7169-95.jpg"
    st.sidebar.image(logo_url, width=200)
    st.sidebar.title("About")
    st.sidebar.markdown("This app predicts house prices in Perth based on a machine learning model. It takes into account the important factors of a house and accurately predicts the price of houses.")
    st.sidebar.markdown("Developed by ")
    st.sidebar.markdown("HAFEEZAH BINTE ABDUL KASIM")
    


    # Load the DataFrame with the 'SUBURB_STATUS_CODE' column
    df3 = pd.read_csv('df3.csv')

    # Get user inputs
    user_inputs = get_user_inputs(df3)

    # Make predictions and display the result
    if st.button("Predict"):
        predicted_price = predict_house_price(user_inputs)
        st.success(f"Predicted House Price: ${round(predicted_price, 2)}")

if __name__ == "__main__":
    main()






# In[ ]:


#(https://emojipedia.org/) icons

