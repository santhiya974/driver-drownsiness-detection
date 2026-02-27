# Driver Monitoring System

## Overview
This project is a **Driver Monitoring System** that detects **drowsiness and mobile phone usage** using **YOLOv8, OpenCV, and Dlib**. The system continuously monitors the driver's facial expressions and actions to ensure road safety.

## Features
- **Drowsiness Detection**: Uses EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) to detect if the driver's eyes are closed or mouth is open for a prolonged period.
- **Phone Usage Detection**: Identifies if the driver is using a mobile phone while driving.
- **Head Tilt Detection**: Determines if the driver's head is tilted abnormally.
- **Alarm System**: Triggers an alert when drowsiness or phone usage is detected.
- **Database Integration**: Sends drowsiness and phone usage events to a **MySQL server** via a **Node.js backend** using **Axios**.

## System Workflow
1. Capture real-time video frames using OpenCV.
2. Detect face and facial landmarks using Dlib.
3. Analyze EAR and MAR values to determine drowsiness.
4. Detect phone usage using YOLOv8 object detection.
5. If drowsiness or phone usage is detected:
   - Start a timer and send the event to the server with a timestamp.
   - If the condition stops, record the end time and send the event duration to the server.
6. Repeat this process frame-by-frame for continuous monitoring.

## Technologies Used
- **Python**: Main programming language.
- **OpenCV**: Image processing and video frame handling.
- **Dlib**: Facial landmark detection.
- **YOLOv8**: Object detection for mobile phone usage.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python

### Python Dependencies
```sh
pip install -r requirements.txt
```

### Running the System
1. Start the Python script:
```sh
python detection.py
```
2. Ensure the Node.js server is running to handle database communication.

## Data Flow Diagram
```text
[Camera] -> [OpenCV & Dlib] -> [YOLOv8 Detection] -> [Condition Check] -> [Send Data to Backend] -> [MySQL Database]
```

## Future Enhancements
- Implement **Real-time Dashboard** for monitoring multiple drivers.
- Improve model accuracy using **custom YOLOv8 training**.
- Deploy as an **Edge AI solution** for in-vehicle devices.


