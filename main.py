import streamlit as st
from src.Plant_Disease.pipelines.prediction_pipeline import predict_image
from PIL import Image
import os

st.title("🌿 Plant Disease Detection")

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type = ["jpg",'jpeg','png']
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img,caption="Uploaded Image",use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            temp_path = "temp_image.jpg"
            img.save(temp_path)

            label,confidence = predict_image(temp_path)

            os.remove(temp_path)

        st.success("Prediction Done ✅")
        st.write(f"🌱 **Disease:** {label}")
        st.write(f"📊 **Confidence:** {confidence} %")
