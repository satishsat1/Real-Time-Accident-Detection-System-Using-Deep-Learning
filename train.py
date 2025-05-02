import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Flatten, Dense, BatchNormalization, MaxPooling3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence

# ✅ Set dataset directory
VIDEO_DATASET_DIR = r"C:\Users\satis\OneDrive\Documents\Final year project IV-2\efficient dataset"

# ✅ Video Data Generator (Efficient RAM Usage)
class VideoDataGenerator(Sequence):
    def __init__(self, dataset_dir, labels={'accident': 1, 'no_accident': 0}, img_size=(224, 224), sequence_length=30, batch_size=4, shuffle=True):
        self.dataset_dir = dataset_dir
        self.labels = labels
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.video_paths, self.video_labels = self._load_video_paths()
        self.on_epoch_end()

    def _load_video_paths(self):
        video_paths = []
        video_labels = []
        for label_name, label in self.labels.items():
            label_dir = os.path.join(self.dataset_dir, label_name)
            if not os.path.exists(label_dir):
                print(f"⚠️ Warning: '{label_dir}' does not exist! Skipping...")
                continue  
            video_files = [f for f in os.listdir(label_dir) if f.endswith(('.mp4', '.avi'))]
            for video_file in video_files:
                video_paths.append(os.path.join(label_dir, video_file))
                video_labels.append(label)
        return video_paths, video_labels

    def __len__(self):
        return int(np.floor(len(self.video_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_paths = self.video_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.video_labels[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self._load_batch(batch_paths, batch_labels)
        return X, y

    def _load_batch(self, batch_paths, batch_labels):
        batch_data = []
        batch_targets = []
        for video_path, label in zip(batch_paths, batch_labels):
            frames = self._preprocess_video(video_path)
            batch_data.append(frames)
            batch_targets.append(label)
        return np.array(batch_data), np.array(batch_targets)

    def _preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, frame_count // self.sequence_length)  # Sample frames evenly
        frames = []
        for i in range(self.sequence_length):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.img_size) / 255.0
            else:
                frame = np.zeros((*self.img_size, 3))  # Padding
            frames.append(frame)
        cap.release()
        return np.array(frames, dtype=np.float32)

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.video_paths, self.video_labels))
            np.random.shuffle(temp)
            self.video_paths, self.video_labels = zip(*temp)

# ✅ Load Data Generator
batch_size = 4  # Adjust to reduce memory usage
train_generator = VideoDataGenerator(VIDEO_DATASET_DIR, batch_size=batch_size)

# ✅ Build a Memory-Efficient Model
def build_model(input_shape=(30, 224, 224, 3)):
    model = Sequential()
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ✅ Train the Model Using Data Generator
model = build_model()
model.fit(train_generator, epochs=10)

# ✅ Save Model
model.save('efficient_i3d_convlstm_model.h5')
print("✅ Model training completed and saved successfully!")
