# Fall Detector

This project is a real-time **fall detection system** built with **Python**, using computer vision and pose estimation technologies. It monitors a person's movements through a webcam feed and detects falls or stumbles based on body keypoint tracking. 
If a fall or stumble is detected, it can automatically send alert messages via WhatsApp.

---

## Features

- **Pose Estimation** using MediaPipe
- **Movement Detection** using background subtraction
- **Face Detection** using Haar cascades
- **Real-Time Fall and Stumble Detection**
- **Automatic WhatsApp Notifications** in case of incidents
- **Visual Feedback** on camera window (labels for movement, stumbles, falls)

---

## Technologies Used

- **Python**
- **OpenCV** — for image processing
- **Mediapipe** — for pose landmark detection
- **NumPy** — for numerical operations
- **pywhatkit** — to send WhatsApp messages automatically
- **Haar Cascade Classifier** — for face detection

---

## Requirements

Make sure you have Python 3 installed. Then install the required libraries:

```bash
pip install opencv-python mediapipe numpy pywhatkit
