import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
st.markdown(
    """
    <style>
    
    .block-container {
        max-width: 800px;
        padding: 20px;
    }
    
]
    </style>
    """,
    unsafe_allow_html=True
)
# Load dataset for Traffic Congestion Prediction
traffic_data = pd.read_csv("Expanded_Traffic_Congestion_Dataset.csv")

# Handle missing values
numerical_features = ['Congestion_Level', 'Accident_Reported']
traffic_data[numerical_features] = traffic_data[numerical_features].apply(pd.to_numeric, errors='coerce')
traffic_data[numerical_features] = traffic_data[numerical_features].fillna(traffic_data[numerical_features].median())

# Normalize data
traffic_data['Location'] = traffic_data['Location'].astype(str).str.strip().str.lower()

# Create and save encoder & scaler if not exist
encoder_traffic = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_traffic.fit(traffic_data[['Location']])
joblib.dump(encoder_traffic, "encoder_traffic.pkl")

scaler = StandardScaler()
scaler.fit(traffic_data[numerical_features])
joblib.dump(scaler, "scaler.pkl")

# Define and train Neural Network model
#X_traffic = np.hstack([encoder_traffic.transform(traffic_data[['Location']]), scaler.transform(traffic_data[numerical_features])])
#y_traffic = np.random.rand(X_traffic.shape[0])  # Placeholder, replace with actual target variable if available

#nn_model = keras.Sequential([
   # keras.layers.Dense(64, activation='relu', input_shape=(X_traffic.shape[1],)),
   # keras.layers.Dense(32, activation='relu'),
   # keras.layers.Dense(16, activation='relu'),
    #keras.layers.Dense(1)
#])

#nn_model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
#nn_model.fit(X_traffic, y_traffic, epochs=10, batch_size=16, verbose=0)
#nn_model.save("traffic_nn_model.h5")

# Load pre-trained models
encoder_traffic = joblib.load("encoder_traffic.pkl")
scaler = joblib.load("scaler.pkl")
nn_model = keras.models.load_model("traffic_nn_model.h5")

# Streamlit UI
st.title("Traffic Congestion Prediction")
location = st.selectbox("Select Location", traffic_data['Location'].unique())
congestion_level = st.selectbox("Congestion Level", list(range(0, 11)))
accident_reported = st.selectbox("Accident Reported?", ["ไม่มี", "มี"])
accident_reported = 1 if accident_reported == "มี" else 0

if st.button("Predict Traffic Congestion"):
    input_data = pd.DataFrame([[location, congestion_level, accident_reported]], columns=['Location', 'Congestion_Level', 'Accident_Reported'])
    input_data['Location'] = input_data['Location'].str.strip().str.lower()
    input_encoded = encoder_traffic.transform(input_data[['Location']])
    input_scaled = scaler.transform(input_data[['Congestion_Level', 'Accident_Reported']])
    input_combined = np.hstack([input_encoded, input_scaled])
    prediction = nn_model.predict(input_combined)[0][0]
    st.write(f"Predicted Traffic Density: {round(prediction, 2)}")
    
    # Generate and display a graph
    st.subheader("Traffic Congestion Trends")
    st.write("กราฟนี้แสดงแนวโน้มอุบัติเหตุตามระดับความแออัดของจราจร โดยแกน X แสดงระดับความแออัดของจราจร และแกน Y แสดงค่าเฉลี่ยของอุบัติเหตุที่เกิดขึ้นในแต่ละระดับความแออัด ช่วยให้เห็นความสัมพันธ์ระหว่างระดับความแออัดและจำนวนอุบัติเหตุ")
    fig, ax = plt.subplots()
    traffic_data.groupby('Congestion_Level')['Accident_Reported'].mean().plot(kind='bar', ax=ax)
    ax.set_xlabel("Congestion Level")
    ax.set_ylabel("Average Accidents")
    ax.set_title("Accident Trends Based on Congestion Level")
    st.pyplot(fig)
