import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Load trained model
model = tf.keras.models.load_model("models/currency_detector_mobilenetv2.h5")

# Define class labels (same order as training)
class_labels = ['10', '20', '50', '100', '200', '500', '2000']

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 170)  # speed of speech
engine.setProperty('volume', 1)  # max volume

# Function to announce currency
def announce_currency(label):
    engine.say(f"This is a {label} rupee note")
    engine.runAndWait()

# Start video capture
cap = cv2.VideoCapture(0)

# For stable prediction — keep track of last prediction
last_prediction = None
last_announce_time = 0

print("🔍 Starting real-time currency detection...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame for prediction
    img = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img / 255.0, axis=0)

    # Get prediction probabilities
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    label_index = np.argmax(predictions)
    label = class_labels[label_index]

    # Show prediction if confidence > threshold (to avoid false positives)
    if confidence > 0.60:
        cv2.putText(frame, f"{label} Rupee ({confidence*100:.1f}%)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # Announce only if different from last prediction or after 5 seconds
        if (label != last_prediction) or (time.time() - last_announce_time > 5):
            announce_currency(label)
            last_prediction = label
            last_announce_time = time.time()
    else:
        cv2.putText(frame, "No currency detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        last_prediction = None

    # Display output
    cv2.imshow("Indian Currency Detector", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Detection stopped.")
