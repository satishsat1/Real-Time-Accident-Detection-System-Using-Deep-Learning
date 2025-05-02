import cv2
import numpy as np
import os
import tensorflow as tf
from twilio.rest import Client
from geopy.geocoders import Nominatim  # To get location name from coordinates

# Load pre-trained model
model = tf.keras.models.load_model('efficient_i3d_convlstm_model.h5')

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

# ✅ Twilio Credentials
account_sid = 'ACb0990872d4642bce684cf6f82be4008e'
auth_token = 'f4b4ae97d4fc3367f1ff53918e09f623'
client = Client(account_sid, auth_token)

# ✅ Function to send SMS Alert
def send_sms(to_number, message_body):
    try:
        message = client.messages.create(
            body=message_body,
            from_='+17622496963',
            to=to_number
        )
        print(f" Message sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f" Failed to send message. Error: {e}")

# ✅ Function to get location from GPS coordinates
def get_location(latitude, longitude):
    geolocator = Nominatim(user_agent="accident_detection_system")
    try:
        location = geolocator.reverse((latitude, longitude), exactly_one=True)
        return location.address if location else "Unknown Location"
    except Exception as e:
        print(f" Error fetching location: {e}")
        return "Unknown Location"

# ✅ Function to predict accident and send an alert
def predict_accident(video_path, latitude=None, longitude=None):
    video_data = preprocess_video(video_path)
    video_data = np.expand_dims(video_data, axis=0)  # Add batch dimension

    prediction = model.predict(video_data)
    accident_probability = prediction[0][0]  # Probability of accident (assuming binary classification)

    if accident_probability > 0.5:  # Threshold for accident detection
        accident_confidence = accident_probability * 100  # Convert to percentage
        location = get_location(latitude, longitude) if latitude and longitude else "Unknown Location"

        print(f" Accident Detected! Confidence: {accident_confidence:.2f}% at {location}")

        recipient_number = '+916305859892'  # Replace with actual emergency number
        alert_message = (f" Emergency Alert \n"
                         f"An accident has been detected with {accident_confidence:.2f}% confidence.\n"
                         f"Location: {location}\n"
                         f"Please respond immediately! ")

        send_sms(recipient_number, alert_message)
    else:
        print(" No Accident Detected.")

# ✅ Example Usage
predict_accident('video_data.mp4', latitude=12.9716, longitude=77.5946)  # Example: Bangalore coordinates
