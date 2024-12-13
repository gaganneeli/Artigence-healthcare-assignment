import streamlit as st
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# Load the fine-tuned YOLOv10 model
model = YOLO(r"C:\Users\ngaga\OneDrive\Desktop\internshala\best.pt")  # Path to your best.pt file

# Streamlit title and description
st.title("YOLOv10 Object Detection")
st.write("Upload an image to detect objects.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Run inference on the uploaded image
    results = model(image)

    # Get the result for the first image
    result = results[0]

    # Plot the results (add bounding boxes to the image)
    annotated_image = result.plot()

    # Display the annotated image in Streamlit
    st.image(annotated_image, caption="Processed Image with Predictions", use_container_width=True)

    # Extract the boxes and class labels
    boxes = result.boxes
    class_ids = boxes.cls.tolist()  # Class IDs
    class_names = [result.names[int(cls_id)] for cls_id in class_ids]  # Class names
    confidence = boxes.conf.tolist()  # Confidence (Precision)

    # Create a DataFrame for the table
    df = pd.DataFrame({
        "Class": class_names,
        "Confidence (Precision)": confidence
    })

    # Display the table with class-wise Precision
    st.write("### Precision for Each Class")
    st.dataframe(df)

    # Calculate overall Precision as the mean of confidence scores
    overall_precision = sum(confidence) / len(confidence) if confidence else 0.0

    # Display overall Precision
    st.write(f"### Overall Precision (mean of all detections): {overall_precision:.4f}")
