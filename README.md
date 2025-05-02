# Accident Detection System using Deep Learning

## Overview
This project is a deep learning-based accident detection system that analyzes surveillance video footage to identify road accidents in real-time. The model uses Conv3D and ConvLSTM2D architectures to classify video segments and sends SMS alerts via Twilio API when an accident is detected.

## Features
- Real-time accident detection from video
- Conv3D + ConvLSTM2D deep learning model
- Video preprocessing pipeline
- Twilio SMS integration for emergency alerts
- Accurate classification of accident vs. non-accident scenes

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Twilio API

## Project Structure
train.py # Model training using a custom data generator
test.py # Evaluation and accuracy on test video data
detection.py # Live prediction and SMS alert system
location code.py # sends location
dataset/ # Accident and non-accident video clips

Requirements
Python 3.7+

TensorFlow

OpenCV

Twilio

NumPy

Dataset
Custom video dataset with accident and non-accident scenes. You can modify the dataset path in the script files.

Output Example
Input: head_on_collision_5.mp4

Output: Detected: Accident
SMS sent to registered number.

Acknowledgments
Thanks to open-source contributors and video datasets that enabled this research.

License
This project is licensed under the MIT License.

tensorflow==2.13.0
keras==2.13.1
opencv-python==4.8.0.74
numpy==1.23.5
twilio==8.4.0
scikit-learn==1.2.2
matplotlib==3.7.1
pandas==1.5.3
To use it:

Save the above content as requirements.txt in your project root.

Install all dependencies by running:
pip install -r requirements.txt

