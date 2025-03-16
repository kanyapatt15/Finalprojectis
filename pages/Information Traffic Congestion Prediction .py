import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Apply custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Itim&display=swap');      

    * {}

    .block-container {
        max-width: 1000px;
        padding: 20px;
    }

    .stApp {
        background-color: #0000; /* เปลี่ยนสีพื้นหลัง */
        color: white; /* เปลี่ยนสีตัวหนังสือ */
    }

    /* ปรับขนาดฟอนต์ให้ใหญ่ขึ้น */
    .custom-header {
        font-size: 45px !important; 
        font-weight: bold;
        text-align: center;
        color: #F6EACB;
    }
    .custom-subheader {
        font-size: 30px !important;
        font-weight: bold;
        color: #F1D3CE;
    }
    .custom-text {
        font-size: 22px !important;
        color: #FFFFFF;
    }
    .footer {
        font-size: 18px;
        text-align: center;
        margin-top: 50px;
        color: #EECAD5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page Content
st.markdown("<p class='custom-header'>📊 ข้อมูล Neural Network และ Feature ของ Dataset</p>", unsafe_allow_html=True)

st.markdown("<p class='custom-subheader'>🔹 แหล่งที่มาของชุดข้อมูล</p>", unsafe_allow_html=True)
st.markdown("<p class='custom-text'>ชุดข้อมูลนี้ถูกสร้างขึ้นโดยใช้ ChatGPT และเป็นข้อมูลจำลองเกี่ยวกับการจราจร</p>", unsafe_allow_html=True)



st.markdown("<p class='custom-subheader'>⚙️ Feature ของ Dataset</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li><b>Location:</b> ตำแหน่งที่เกิดการจราจรติดขัด</li>
        <li><b>Time:</b> เวลาที่บันทึกข้อมูลจราจร</li>
        <li><b>Accident_Reported:</b> รายงานอุบัติเหตุ (0 = ไม่มี, 1 = มี)</li>
        <li><b>Weather_Condition:</b> สภาพอากาศ เช่น ฝนตก หมอก แดดจ้า</li>
        <li><b>Traffic_Density:</b> ความหนาแน่นของจราจร (เป้าหมายสำหรับการพยากรณ์)</li>
    </ul>
    """,
    unsafe_allow_html=True
)
st.markdown("<p class='custom-subheader'>📌 ทฤษฎีของอัลกอริทึม Neural Network</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li><b>Neural Network:</b> Neural Network เป็นโมเดลที่เลียนแบบโครงสร้างของสมองมนุษย์ โดยใช้ชั้นของเซลล์ประสาท (Neurons) หลายชั้น ซึ่งสามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนได้</li>
        <ul class='custom-text'>
            <li><b>ชั้น Input Layer:</b> รับค่าข้อมูลที่เข้ามา เช่น ตำแหน่งการจราจร, เวลา, สภาพอากาศ</li>
            <li><b>ชั้น Hidden Layers:</b> ประมวลผลข้อมูลผ่านการเชื่อมโยงแบบไม่เชิงเส้นโดยใช้ Activation Functions เช่น ReLU</li>
            <li><b>ชั้น Output Layer:</b> ทำนายค่าความแออัดของจราจร</li>
        </ul>
    </ul>
    """,
    unsafe_allow_html=True
)
st.markdown("<p class='custom-subheader'>🛠️ ขั้นตอนการพัฒนาโมเดล Neural Network</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li>ตรวจสอบค่าที่หายไปและแทนที่ด้วยค่ามัธยฐาน</li>
        <li>แปลงค่าหมวดหมู่โดยใช้ One-Hot Encoding</li>
        <li>ปรับขนาดข้อมูลด้วย StandardScaler</li>
        <li>แบ่งข้อมูลเป็น 80% ชุดฝึก และ 20% ชุดทดสอบ</li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='custom-subheader'>🤖 การฝึกสอน Neural Network</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li>สร้างโมเดลโดยใช้ keras.Sequential()</li>
        <li>เพิ่ม 3 Hidden Layers:
            <ul class='custom-text'>
                <li>Layer 1: Dense(64, activation='relu')</li>
                <li>Layer 2: Dense(32, activation='relu')</li>
                <li>Layer 3: Dense(16, activation='relu')</li>
            </ul>
        </li>
        <li>ชั้น Output: Dense(1) สำหรับพยากรณ์ค่าความหนาแน่น</li>
        <li>ใช้ optimizer='adam' และ loss='mean_squared_error'</li>
        <li>ฝึกโมเดลด้วย 10 epochs และ batch_size=16</li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='custom-subheader'>📈 การประเมินผลและการใช้งานโมเดล</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li>ทดสอบโมเดลด้วยชุดทดสอบและวัดผลด้วย MSE และ MAE</li>
        <li>ใช้งานโมเดลในแอปพลิเคชันเพื่อทำนายค่าความหนาแน่นจราจรตามปัจจัยต่าง ๆ</li>
    </ul>
    """,
    unsafe_allow_html=True
)

# Model comparison
st.markdown("<p class='custom-subheader'>📊 เปรียบเทียบโมเดล</p>", unsafe_allow_html=True)
model_comparison = pd.DataFrame({
    "โมเดล": ["Linear Regression", "Extra Trees", "Random Forest", "Neural Network"],
    "ข้อดี": [
        "ใช้งานง่ายและตีความได้ง่าย",
        "สามารถลด Overfitting ได้ดี",
        "มีความแม่นยำสูงในข้อมูลที่ซับซ้อน",
        "เรียนรู้ความสัมพันธ์ที่ซับซ้อนได้ดี"
    ],
    "ข้อเสีย": [
        "ไม่สามารถจัดการความสัมพันธ์ที่ซับซ้อนได้ดี",
        "ต้องการพลังคำนวณสูงเมื่อข้อมูลมาก",
        "ใช้เวลาฝึกนาน",
        "ต้องใช้ข้อมูลจำนวนมากในการฝึก"
    ]
})
st.markdown(model_comparison.to_html(index=False), unsafe_allow_html=True)

# Footer
st.markdown("<p class='footer'>จัดทำโดย นางสาวกัญญาพัชร ก้อนนิล | รหัสนักศึกษา: 6404062620125</p>", unsafe_allow_html=True)

