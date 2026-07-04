import streamlit as st
from PIL import Image
import os
from src import ObjectDetector


@st.cache_resource
def load_detector():
    return ObjectDetector()

detector = load_detector()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Object Detector", page_icon="🐦")

st.title("🐦 Object Detector")
st.write("Upload an image to see what object YOLO detects with the highest confidence!")

# 1. Image Upload Widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display the uploaded image nicely on the web page
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("🔄 *Running object detection...*")
    
    # Temporary save path because YOLO functions look for a file path/directory
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    try:
        # 2. Run the YOLO algorithm using your first class
        raw_result = detector.load_and_predict(temp_path)
        
        # 3. Extract the highest object prediction and confidence score
        top_class_name, confidence = detector.evaluate_prediction(raw_result)
        
        # Display the results in clean UI metric cards
        st.success("🎉 Detection Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Detected Object", value=top_class_name.title())
        with col2:
            st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
            
    finally:
        # Clean up the temporary file so we don't litter your repository
        if os.path.exists(temp_path):
            os.remove(temp_path)