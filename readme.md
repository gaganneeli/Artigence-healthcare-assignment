# YOLOv10 Object Detection with Streamlit

This project demonstrates the use of the fine-tuned YOLOv10 model for object detection in a Streamlit web app. Users can upload images to the app, and the model will detect objects, display bounding boxes, and show class-wise precision (confidence).

## Features

- Upload images in JPG, JPEG, or PNG format.
- YOLOv10 model for object detection.
- Bounding boxes drawn around detected objects.
- Displays the object classes and their confidence (precision) scores.
- Calculates overall precision as the mean of confidence scores.

## Online Demo

You can try out the object detection app directly online by visiting the following link:

[YOLOv10 Object Detection Demo](https://huggingface.co/spaces/gaganneeli/YOLOv10_on_BCCD)

## Requirements

Before running this app, ensure you have the following dependencies installed:

- Python 3.x
- Streamlit
- ultralytics
- Pillow
- Matplotlib
- pandas

You can install the required dependencies using pip:

```bash
pip install streamlit ultralytics Pillow matplotlib pandas
