import cv2
import numpy as np
import os
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPooling3D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split


VIDEO_DATASET_DIR = r"C:\Users\satis\OneDrive\Documents\Final year project IV-2\testing dataset\testing dataset"

def preprocess_video(video_path, img_size=(224, 224), sequence_length=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends
        
        resized_frame = cv2.resize(frame, img_size)
        normalized_frame = resized_frame / 255.0  # Normalize pixel values (0-1)
        frames.append(normalized_frame)

    cap.release()

    # ✅ Pad with black frames if video is too short
    while len(frames) < sequence_length:
        frames.append(np.zeros((img_size[0], img_size[1], 3)))

    return np.array(frames)

# ✅ Function to load all videos from a directory and process them
def process_video_dataset(dataset_dir, sequence_length=30, img_size=(224, 224)):
    video_data = []
    labels = []

    for label in ['accident', 'no_accident']:
        label_dir = os.path.join(dataset_dir, label)

        # ✅ FIX: Debugging step to check directory existence
        if not os.path.exists(label_dir):
            print(f"❌ Error: Directory '{label_dir}' does not exist! Skipping...")
            continue  

        video_files = [f for f in os.listdir(label_dir) if f.endswith(('.mp4', '.avi'))]

        # ✅ FIX: Check if dataset contains videos
        if len(video_files) == 0:
            print(f"⚠️ Warning: No videos found in '{label_dir}'!")
            continue  

        for video_file in video_files:
            video_path = os.path.join(label_dir, video_file)
            print(f"✅ Processing video: {video_file} ({label})")
            frames = preprocess_video(video_path, img_size=img_size, sequence_length=sequence_length)
            video_data.append(frames)
            labels.append(1 if label == 'accident' else 0)

    return np.array(video_data), np.array(labels)

# ✅ Load dataset
X, y = process_video_dataset(VIDEO_DATASET_DIR)

import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('efficient_i3d_convlstm_model.h5')

# Display the loaded model architecture
model.summary()

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')