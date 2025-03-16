import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset for Metro Delay Prediction
metro_data = pd.read_csv("Expanded_Metro_Delay_Dataset.csv")

# Handle missing values
metro_data['Delay_Minutes'].fillna(metro_data['Delay_Minutes'].median(), inplace=True)
metro_data['Reason'].fillna("unknown", inplace=True)

# Normalize data
metro_data['Station'] = metro_data['Station'].astype(str).str.strip().str.lower()
metro_data['Reason'] = metro_data['Reason'].astype(str).str.strip().str.lower()

# Extract hour from time feature
metro_data['Hour'] = pd.to_datetime(metro_data['Time'], format='%H:%M').dt.hour
metro_data.drop(columns=['Time'], inplace=True)

# One-Hot Encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_features = metro_data[['Station', 'Reason']]
categorical_encoded = encoder.fit_transform(categorical_features)
categorical_columns = encoder.get_feature_names_out(['Station', 'Reason'])
categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_columns)

# Combine encoded features with numerical features
X = pd.concat([categorical_df, metro_data[['Hour']]], axis=1)
y = metro_data['Delay_Minutes']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "metro_rf_model.pkl")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, "metro_lr_model.pkl")

et_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)
joblib.dump(et_model, "metro_et_model.pkl")

# Load pre-trained models
encoder = joblib.load("encoder.pkl")
rf_model = joblib.load("metro_rf_model.pkl")
lr_model = joblib.load("metro_lr_model.pkl")
et_model = joblib.load("metro_et_model.pkl")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Itim&display=swap');      

    .block-container {
        max-width: 800px;
        padding: 20px;
    }
  
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit UI
st.title("Metro Delay Prediction System")

st.header("Metro Delay Prediction")
station = st.selectbox("Select Station", metro_data['Station'].unique())
time_metro = st.text_input("Enter Time (HH:MM)", "08:00")
reason = st.selectbox("Select Reason", metro_data['Reason'].unique())
model_choice = st.selectbox("Select Model", ["Random Forest", "Linear Regression", "Extra Trees"])

if st.button("Predict Metro Delay"):
    input_data = pd.DataFrame([[station, reason]], columns=['Station', 'Reason'])
    input_encoded = encoder.transform(input_data)
    input_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['Station', 'Reason']))
    input_df['Hour'] = pd.to_datetime(time_metro, format='%H:%M').hour
    
    predictions = {
        "Random Forest": rf_model.predict(input_df)[0],
        "Linear Regression": lr_model.predict(input_df)[0],
        "Extra Trees": et_model.predict(input_df)[0]
    }
    
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(list(predictions.items()), columns=["Model", "Predicted Delay (minutes)"])
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(list(predictions.items()), columns=["Model", "Predicted Delay (minutes)"])
    
    if model_choice == "Random Forest":
        prediction = rf_model.predict(input_df)[0]
    elif model_choice == "Linear Regression":
        prediction = lr_model.predict(input_df)[0]
    else:
        prediction = et_model.predict(input_df)[0]
    
    st.write(f"Predicted Metro Delay: {round(prediction, 2)} minutes")
    
    # Display table of predictions
    st.subheader("Comparison of Model Predictions")
    st.write("ตารางนี้แสดงค่าความล่าช้าที่คาดการณ์โดยโมเดลต่าง ๆ เพื่อให้สามารถเปรียบเทียบความแตกต่างของผลลัพธ์")
    st.dataframe(pred_df)
    input_data = pd.DataFrame([[station, reason]], columns=['Station', 'Reason'])
    input_encoded = encoder.transform(input_data)
    input_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['Station', 'Reason']))
    input_df['Hour'] = pd.to_datetime(time_metro, format='%H:%M').hour
    
    
    # Generate and display a graph
    st.subheader("Metro Delay Trends by Hour")
    st.write("กราฟนี้แสดงค่าเฉลี่ยของความล่าช้าของรถไฟใต้ดินในแต่ละชั่วโมงของวัน ช่วยให้สามารถระบุช่วงเวลาที่มีความล่าช้าสูงสุดและเข้าใจแนวโน้มของความล่าช้าได้ดีขึ้น")
    fig, ax = plt.subplots()
    metro_data.groupby('Hour')['Delay_Minutes'].mean().plot(kind='bar', ax=ax)
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Average Delay (minutes)")
    ax.set_title(f"Average Metro Delay by Hour for {station.capitalize()}")
    st.pyplot(fig)
