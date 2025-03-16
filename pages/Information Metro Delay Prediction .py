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

    * {
        
    }

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
st.markdown("<p class='custom-header'>📊 ข้อมูล Model และ Feature ของ Dataset</p>", unsafe_allow_html=True)

st.markdown("<p class='custom-subheader'>🔹 แหล่งที่มาของชุดข้อมูล</p>", unsafe_allow_html=True)
st.markdown("<p class='custom-text'>ชุดข้อมูลนี้ถูกสร้างขึ้นโดยใช้ ChatGPT และเป็นข้อมูลจำลองเกี่ยวกับความล่าช้าของรถไฟใต้ดิน</p>", unsafe_allow_html=True)

st.markdown("<p class='custom-subheader'>📌 คุณลักษณะของชุดข้อมูล</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li><b>Station:</b> สถานีรถไฟใต้ดินที่มีการบันทึกความล่าช้า</li>
        <li><b>Reason:</b> เหตุผลของความล่าช้า (เช่น ปัญหาทางเทคนิค, สภาพอากาศ, ความแออัด)</li>
        <li><b>Hour:</b> ช่วงเวลาที่เกิดความล่าช้าในแต่ละวัน</li>
        <li><b>Delay_Minutes:</b> จำนวนเวลาที่รถไฟใต้ดินล่าช้า (นาที)</li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='custom-subheader'>⚙️ กระบวนการเตรียมข้อมูล</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li>จัดการค่าที่หายไปโดยแทนที่ด้วยค่ามัธยฐาน</li>
        <li>แปลงค่าข้อมูลเชิงหมวดหมู่โดยใช้ One-Hot Encoding</li>
        <li>ปรับขนาดค่าข้อมูลเชิงตัวเลขโดยใช้ StandardScaler</li>
    </ul>
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='custom-subheader'>🤖 ทฤษฎีของอัลกอริทึมที่พัฒนา</p>", unsafe_allow_html=True)
st.markdown(
    """
    <ul class='custom-text'>
        <li><b>Random Forest:</b> อัลกอริทึมที่ใช้การสร้างหลายต้นไม้การตัดสินใจ (Decision Trees) และเฉลี่ยผลลัพธ์เพื่อลด Overfitting</li>
        <li><b>Linear Regression:</b> โมเดลที่ใช้สมการเชิงเส้นเพื่อหาความสัมพันธ์ระหว่างตัวแปรอิสระกับค่าความล่าช้า</li>
        <li><b>Extra Trees:</b> อัลกอริทึมที่คล้ายกับ Random Forest แต่มีการสุ่มจุดแบ่งข้อมูลมากขึ้น</li>
    </ul>
    """,
    unsafe_allow_html=True
)

# เปรียบเทียบโมเดล
st.markdown("<p class='custom-subheader'>📈 เปรียบเทียบโมเดลที่ใช้</p>", unsafe_allow_html=True)
model_comparison = pd.DataFrame({
    "โมเดล": ["Random Forest", "Linear Regression", "Extra Trees"],
    "ข้อดี": [
        "สามารถจับความสัมพันธ์ที่ซับซ้อนได้ดีและลดการเกิด overfitting",
        "โมเดลง่ายและรวดเร็ว ใช้ตีความได้ง่าย",
        "สามารถให้ผลลัพธ์ที่แม่นยำและทนต่อค่า outliers"
    ],
    "ข้อเสีย": [
        "ต้องใช้เวลาในการฝึกโมเดลมากกว่ารุ่นที่ง่ายกว่า",
        "มีข้อจำกัดเมื่อต้องจับความสัมพันธ์ที่ซับซ้อน",
        "ต้องการพลังคำนวณมากขึ้นเมื่อมีข้อมูลจำนวนมาก"
    ]
})
st.markdown(model_comparison.to_html(index=False), unsafe_allow_html=True)

st.markdown("<p class='custom-subheader'>🛠️ ขั้นตอนการพัฒนาโมเดล</p>", unsafe_allow_html=True)

st.markdown("<p class='custom-subheader'>1. ขั้นตอนการพัฒนาโมเดล Random Forest</p>", unsafe_allow_html=True)
st.markdown(
    """
    💡 แนวคิดของโมเดล:
    Random Forest เป็นอัลกอริทึมที่ใช้หลายต้นไม้การตัดสินใจ (Decision Trees) ทำงานร่วมกัน
    โดยรวมผลลัพธ์ของแต่ละต้นไม้เพื่อลด Overfitting และเพิ่มความแม่นยำในการพยากรณ์
    
    1️⃣ การเตรียมข้อมูล
    ตรวจสอบค่าที่ขาดหาย (Missing Values) และแทนค่าที่ขาดหายด้วยค่ามัธยฐาน (Median)
    แปลงค่าข้อมูลที่เป็นหมวดหมู่ (เช่น ชื่อสถานี, สาเหตุของความล่าช้า) ให้เป็นตัวเลขโดยใช้ One-Hot Encoding
    ปรับสเกลค่าข้อมูลเชิงตัวเลขโดยใช้ StandardScaler
    
    2️⃣ การแบ่งชุดข้อมูล
    แบ่งข้อมูลเป็น 80% สำหรับฝึกโมเดล (Training Set) และ 20% สำหรับทดสอบ (Testing Set)
    
    3️⃣ การฝึกโมเดล
    ใช้ RandomForestRegressor() จาก sklearn.ensemble
    กำหนด n_estimators = 100 เพื่อใช้ต้นไม้การตัดสินใจ 100 ต้น
    ตั้งค่า random_state=42 เพื่อให้ได้ผลลัพธ์ที่คงที่
    ฝึกโมเดลด้วยข้อมูลชุดฝึก (fit(X_train, y_train))
    
    4️⃣ การทดสอบและประเมินผล
    ใช้ predict(X_test) เพื่อพยากรณ์ข้อมูลในชุดทดสอบ
    วัดความแม่นยำของโมเดลด้วย Mean Squared Error (MSE) และ Mean Absolute Error (MAE)
    
    
    """,
    unsafe_allow_html=True
)

st.markdown("<p class='custom-subheader'>2. ขั้นตอนการพัฒนาโมเดล Linear Regression</p>", unsafe_allow_html=True)
st.markdown(
    """
    💡 แนวคิดของโมเดล:
    Linear Regression ใช้สมการเชิงเส้นเพื่อทำนายค่าผลลัพธ์ เหมาะกับข้อมูลที่มีแนวโน้มเป็นเส้นตรง
    
   
    1️⃣ การเตรียมข้อมูล
    ใช้วิธีเดียวกับ Random Forest (เติมค่าที่ขาดหาย + One-Hot Encoding + StandardScaler)
    
    2️⃣ การแบ่งชุดข้อมูล
    แบ่งข้อมูลเป็น 80% Training / 20% Testing
    
    3️⃣ การฝึกโมเดล
    ใช้ LinearRegression() จาก sklearn.linear_model
    ฝึกโมเดลโดยใช้คำสั่ง fit(X_train, y_train)
    
    4️⃣ การทดสอบและประเมินผล
    ใช้ predict(X_test) เพื่อพยากรณ์
    ใช้ r2_score() วัดค่าความแม่นยำของโมเดล
   """,unsafe_allow_html=True)
st.markdown("<p class='custom-subheader'>3. ขั้นตอนการพัฒนาโมเดล Extra Trees Regressor</p>", unsafe_allow_html=True)
st.markdown(
    """
    💡 แนวคิดของโมเดล:
    Extra Trees Regressor เป็นอัลกอริทึมที่คล้ายกับ Random Forest
    แต่มีการสุ่มจุดแบ่งข้อมูล (Split Points) ในแต่ละต้นไม้มากขึ้น ช่วยลด Overfitting ได้ดีกว่า
    
    1️⃣ การเตรียมข้อมูล
    ใช้วิธีเดียวกับโมเดลอื่น (เติมค่าที่ขาดหาย + One-Hot Encoding + StandardScaler)
    
    2️⃣ การแบ่งชุดข้อมูล
    แบ่งข้อมูลเป็น 80% Training / 20% Testing
    
    3️⃣ การฝึกโมเดล
    ใช้ ExtraTreesRegressor() จาก sklearn.ensemble
    ตั้งค่า n_estimators=100 (ใช้ 100 ต้นไม้) และ random_state=42
    ฝึกโมเดลโดยใช้คำสั่ง fit(X_train, y_train)
    
    4️⃣ การทดสอบและประเมินผล
    ใช้ predict(X_test) เพื่อพยากรณ์
    ใช้ Mean Absolute Error (MAE) และ Mean Squared Error (MSE) วัดค่าความผิดพลาดของโมเดล
    
    """,
    unsafe_allow_html=True
)


# Footer - ผู้จัดทำ
st.markdown("<p class='footer'>จัดทำโดย นางสาวกัญญาพัชร ก้อนนิล | รหัสนักศึกษา: 6404062620125</p>", unsafe_allow_html=True)
