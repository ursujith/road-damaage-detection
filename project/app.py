from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import streamlit as st
from PIL import Image


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
            margin:0;
            padding:0;
            box-sizing:border-box;
            font-family: 'Montserrat', sans-serif;
             <!--background-image: url("https://images7.alphacoders.com/101/1018019.jpg");-->
             background-attachment: fixed;
             animation:animateBg 4s linear infinite;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.title("ROAD DAMAGE DETECTION USING YOLOV8")
st.markdown('<style>h1{color: white;font-size:40px;}</style>', unsafe_allow_html=True)
st.markdown("<span style='font-size:20px;color:white;'>Automated Road Damage Detection represents a promising approach to enhancing road maintenance practices through automation, data-driven decision-making, and technological innovation. By leveraging the power of AI and computer vision, ARDD systems have the potential to revolutionize how transportation agencies monitor, assess, and maintain road networks, ultimately leading to safer and more sustainable transportation infrastructure.</span>",unsafe_allow_html=True)
def uploading():
    upload = st.file_uploader("Choose a file")
    if upload is not None:
        st.write(upload.name)
        file_path = os.path.join("results", upload.name)
        with open(file_path, "wb") as user_file:
            user_file.write(upload.getbuffer())
        return file_path, upload.name
    else:
        return None, None

img_path, name = uploading()

if img_path is not None and name is not None:
    model = YOLO("best.pt")
    predictions = model.predict(source=img_path, save=True, save_crop=True, project="output", name="inference", exist_ok=True, save_txt=True)
    print(predictions)
    col1, col2 = st.columns(2)

    with col1:
        st.image(img_path, caption='Uploaded image', width=350)

    with col2:
        st.image('output//inference//' + name, caption='Predicted image', width=350)
    
