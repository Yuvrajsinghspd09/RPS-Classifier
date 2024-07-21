# RPS-Classifier

This project implements a deep learning model to classify images of hand gestures into Rock, Paper, or Scissors categories.
Overview
The classifier uses a pre-trained Keras model to predict the class of a given image. It can distinguish between three classes:

Rock
Paper
Scissors

Requirements

Python 3.x
OpenCV (cv2)
NumPy
TensorFlow
Keras
Matplotlib

Usage

Ensure you have the required libraries installed.
Place your trained model file (best_rps_model.h5) in the same directory as the script.
Update the custom_image_path variable with the path to your test image.
Run the script to see the prediction and visualization.

How it works

The script loads a pre-trained model from best_rps_model.h5.
It preprocesses the input image by resizing it to 224x224 pixels and normalizing the pixel values.
The model then predicts the class of the image.
Finally, it displays the image along with the predicted class and confidence score.

Try it yourself!
You can try out this Rock Paper Scissors classifier online! Visit our demo page to upload your own images and see the model in action.
(Note: Replace the "#" in the link above with the actual URL where you host your model demo)
Contributing
Feel free to fork this repository and submit pull requests to contribute to this project. You can also open issues if you find any bugs or have suggestions for improvements.
