🖐️✊✌️ Rock Paper Scissors Image Classifier

This project implements a deep learning model to classify images of hand gestures into Rock, Paper, or Scissors categories.
🔍 Overview
The classifier uses a pre-trained Keras model to predict the class of a given image. It can distinguish between three classes:

Rock 🪨
Paper 📄
Scissors ✂️

🛠️ Requirements

Python 3.x
Libraries:

OpenCV (cv2)
NumPy
TensorFlow
Keras
Matplotlib



🚀 Usage

Setup: Ensure you have the required libraries installed.
Model Placement: Place your trained model file (best_rps_model.h5) in the same directory as the script.
Image Selection: Update the custom_image_path variable with the path to your test image.
Execution: Run the script to see the prediction and visualization.

🧠 How it works

Model Loading: The script loads a pre-trained model from best_rps_model.h5.
Image Preprocessing:

Resizes the input image to 224x224 pixels
Normalizes the pixel values


Prediction: The model predicts the class of the image.
Visualization: Displays the image with the predicted class and confidence score.

🖥️ Code Breakdown
Key Functions:

preprocess_custom_image(image_path)

Loads and preprocesses the image for model input


predict_custom_image(image_path)

Makes a prediction on the given image
Returns the predicted label and confidence score



Main Execution:
pythonCopy# Load model
model = load_model('best_rps_model.h5')

# Make prediction
label, confidence = predict_custom_image(custom_image_path)

# Display results
print(f"Predicted Class: {label}")
print(f"Confidence: {confidence:.2f}")

# Visualize
plt.imshow(image)
plt.title(f"Prediction: {label} ({confidence:.2f})")
plt.show()

🎮 Try it yourself!
Experience the Rock Paper Scissors classifier in action!

👉 Click here to try our online demo
Upload your own images and watch the AI make its predictions!
(Note: Replace the "#" with your actual demo URL)

🤝 Contributing
We welcome contributions! Here's how you can help:

Fork the repository
Create a new branch for your feature
Commit your changes
Push to the branch
Create a new Pull Request

Found a bug or have a suggestion? Open an issue and let us know!
📄 License
[Include your chosen license here]

Happy classifying! May the best hand win! 🏆
