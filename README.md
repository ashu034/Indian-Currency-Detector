# Indian-Currency-Detector

🇮🇳 Indian Currency Detection System using AI
📌 Project Overview

This project is an AI-powered Indian Currency Detection System that identifies Indian banknotes in real time using a webcam and announces the detected denomination through voice output. The system is designed to detect common Indian currency notes and intelligently avoid false detection when no currency is present.

The project leverages Deep Learning, Computer Vision, and Transfer Learning to achieve accurate and efficient detection.

🚀 Features

Real-time currency detection using webcam

Supports Indian denominations: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000

Voice announcement of detected currency using text-to-speech

Ignores background when no currency note is present

High accuracy using a pre-trained MobileNetV2 model

Lightweight and fast inference suitable for real-time use

🧠 Technologies Used

Python

TensorFlow / Keras

MobileNetV2 (Transfer Learning)

OpenCV

NumPy

pyttsx3 (Text-to-Speech)

Kaggle Indian Currency Dataset

⚙️ How It Works

Currency images are trained using a pre-trained MobileNetV2 CNN.

The trained model extracts visual features of currency notes.

Live webcam frames are processed and classified in real time.

If a valid currency is detected with sufficient confidence, the denomination is displayed and announced using audio.

When no note is present, the system correctly displays “No currency detected”.

▶️ How to Run

Clone the repository

Create and activate a virtual environment

Install dependencies

Train the model using train_model.py

Run real-time detection using realtime_detection.py

🎯 Applications

Assistive technology for visually impaired users

Smart currency recognition systems

AI-based fintech and automation projects

Computer Vision learning projects
