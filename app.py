import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import mediapipe as mp

# Caching the model load to improve performance
@st.cache_resource
def load_rps_model():
    return load_model(r'C:\Users\yuvra\OneDrive\Desktop\RPS GAME\best_rps_model.h5')

model = load_rps_model()

# Class labels
class_labels = ['Paper', 'Rock', 'Scissors']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def detect_and_classify(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = get_bounding_box(hand_landmarks, frame.shape)
            hand_region = frame[y_min:y_max, x_min:x_max]
            
            if hand_region.size != 0:  # Check if hand_region is not empty
                preprocessed = preprocess_image(hand_region)
                prediction = model.predict(preprocessed)
                class_idx = np.argmax(prediction)
                label = class_labels[class_idx]
                confidence = prediction[0][class_idx]
                
                # Draw the bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return frame

def get_bounding_box(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        x_min = max(0, min(x_min, x))
        y_min = max(0, min(y_min, y))
        x_max = min(w, max(x_max, x))
        y_max = min(h, max(y_max, y))
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    
    return x_min, y_min, x_max, y_max

def main():
    st.title("Rock-Paper-Scissors Classifier")
    
    # Add a slider for confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    
    # Add a button to start/stop the webcam
    run = st.checkbox('Start Webcam')
    
    FRAME_WINDOW = st.image([])
    video_capture = cv2.VideoCapture(0)

    while run:
        ret, frame = video_capture.read()
        if not ret:
            st.write("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame = detect_and_classify(frame)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        FRAME_WINDOW.image(rgb_frame)

    video_capture.release()

if __name__ == "__main__":
    main()