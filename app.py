import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Apply custom styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Itim&display=swap');      

    * {
        font-family: 'Itim', cursive !important;
    }
    
    .block-container {
        max-width: 1200px;
        padding: 20px;
    }

    .stApp {
        background-color: #222831; /* เปลี่ยนสีพื้นหลัง */
        color: white; /* เปลี่ยนสีตัวหนังสือ */
    }

    /* ปรับขนาดฟอนต์ให้ใหญ่ขึ้น */
    .custom-title {
        font-size: 50px !important; /* หัวข้อใหญ่ขึ้น */
        font-weight: bold;
        text-align: center;
        color: #F6EACB;
    }
    .custom-subheader {
        font-size: 30px !important; /* หัวข้อรอง */
        font-weight: bold;
        color: #F1D3CE;
    }
    .custom-text {
        font-size: 22px !important; /* ขนาดข้อความทั่วไป */
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

# Home Page
st.markdown("<p class='custom-title'>🚦 ยินดีต้อนรับสู่ระบบพยากรณ์สภาพจราจรและความล่าช้าของรถไฟใต้ดิน 🚆</p>", unsafe_allow_html=True)

st.markdown("<p class='custom-subheader'>🔍 ฟีเจอร์ของระบบ</p>", unsafe_allow_html=True)
st.markdown("<p class='custom-text'>1. พยากรณ์ความแออัดของการจราจร: คาดการณ์ระดับความแออัดของการจราจรแบบเรียลไทม์โดยใช้ข้อมูลสถานที่ อุบัติเหตุ และข้อมูลในอดีต</p>", unsafe_allow_html=True)
st.markdown("<p class='custom-text'>2. พยากรณ์ความล่าช้าของรถไฟใต้ดิน: วิเคราะห์และพยากรณ์ความล่าช้าของรถไฟใต้ดินโดยใช้ข้อมูลในอดีตที่เกี่ยวข้องกับสถานีต่าง ๆ</p>", unsafe_allow_html=True)

# Footer - ผู้จัดทำ
st.markdown("<p class='footer'>จัดทำโดย นางสาวกัญญาพัชร ก้อนนิล  รหัสนักศึกษา: 6404062620125</p>", unsafe_allow_html=True)
