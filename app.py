import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

# Define the feature columns (ensure these match your training data)
feature_columns = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                   'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                   'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                   'Warehouse_Processing_Time']

st.set_page_config(page_title="Delivery Delay Predictor", layout="centered")

st.title("🚚 Delivery Delay Prediction")
st.write("Enter the features below to predict if a delivery will be delayed.")

# Create input fields for each feature
with st.form("prediction_form"):
    st.header("Delivery Details")
    delivery_distance = st.slider("Delivery Distance (km)", 1.0, 100.0, 20.0)
    traffic_congestion = st.selectbox("Traffic Congestion (1=Low, 5=High)", options=[1, 2, 3, 4, 5])
    weather_condition = st.selectbox("Weather Condition (1=Good, 5=Severe)", options=[1, 2, 3, 4, 5])
    delivery_slot = st.selectbox("Delivery Slot (1=Morning, 2=Afternoon, 3=Evening)", options=[1, 2, 3])
    driver_experience = st.slider("Driver Experience (Years)", 1, 20, 5)
    num_stops = st.slider("Number of Stops", 1, 15, 3)
    vehicle_age = st.slider("Vehicle Age (Years)", 1, 10, 3)
    road_condition_score = st.slider("Road Condition Score (1=Poor, 5=Excellent)", 1, 5, 3)
    package_weight = st.slider("Package Weight (kg)", 0.1, 50.0, 5.0)
    fuel_efficiency = st.slider("Fuel Efficiency (km/l)", 5.0, 25.0, 15.0)
    warehouse_processing_time = st.slider("Warehouse Processing Time (minutes)", 10, 120, 60)

    submitted = st.form_submit_button("Predict Delivery Delay")

    if submitted:
        input_data = {
            'Delivery_Distance': delivery_distance,
            'Traffic_Congestion': traffic_congestion,
            'Weather_Condition': weather_condition,
            'Delivery_Slot': delivery_slot,
            'Driver_Experience': driver_experience,
            'Num_Stops': num_stops,
            'Vehicle_Age': vehicle_age,
            'Road_Condition_Score': road_condition_score,
            'Package_Weight': package_weight,
            'Fuel_Efficiency': fuel_efficiency,
            'Warehouse_Processing_Time': warehouse_processing_time
        }
        
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Results")
        if prediction[0] == 1:
            st.error("**Prediction: DELAYED**")
        else:
            st.success("**Prediction: ON TIME**")
