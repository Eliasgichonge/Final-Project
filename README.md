# 🚗 Lane Departure Detection System using OpenCV and Kalman Filter

A real-time lane departure warning system built with Python, OpenCV, Kalman filtering, and an ESP32 camera. This system detects white lane markings, tracks their positions over time using a Kalman filter, and triggers an audio alarm when the vehicle veers out of its lane.

---

## 📌 Features

- 📷 **Real-Time Video Streaming** from ESP32-CAM module.
- 🛣️ **Lane Detection** using multi-color space thresholding and edge detection.
- 📈 **Lane Position Smoothing** with Kalman Filter for stability.
- 🚨 **Audio Alarm** system using `pygame` for lane departure warnings.
- 🧠 **Multiple Detection Techniques**: Histogram analysis, contours, Hough lines.
- 🔁 **Violation Tracking** to reduce false positives.

---

## 🧰 Technologies Used

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pygame (for sound alarm)
- Kalman Filter (custom implementation)
- ESP32-CAM for live video stream

---

## 🔧 Setup Instructions

### 1. Prerequisites

Install the required Python libraries:

```bash
pip install opencv-python numpy pygame
````

Ensure your ESP32-CAM is streaming video and reachable over a local IP (e.g., `http://192.168.192.102:81/stream`).

---

### 2. Running the Project

Update the IP address of your ESP32 camera in `main()`:

```python
CAMERA_IP = "192.168.192.102"
CAMERA_PORT = 81
```

Then run the script:

```bash
python lane_departure_detector.py
```

---

## 🎯 How It Works

1. **Frame Capture**: Captures video frames from the ESP32-CAM stream.
2. **Preprocessing**: Extracts the Region of Interest (ROI) and applies white color detection in HSV, RGB, LAB, and grayscale.
3. **Lane Detection**: Combines histogram peaks, contours, and Hough lines to find left and right lane lines.
4. **Kalman Filter**: Smoothens the detected positions over time to reduce noise.
5. **Lane Departure Check**: Compares the car's center with lane center; triggers violation count and alarm if deviation is large.
6. **Alarm System**: Plays an audio beep repeatedly when departure is detected.

---

## 🧪 Visualization

* **Green lines**: Detected lane boundaries.
* **Yellow line**: Estimated lane center.
* **Purple shaded area**: Danger zones (outside lane).
* **Text overlay**: Shows departure status and violation count.

---

## 📸 ESP32-CAM Setup Tips

Make sure your ESP32-CAM is flashed with the correct firmware and streaming over HTTP. The format should be:

```
http://<CAMERA_IP>:<PORT>/stream
```

You can test the stream with a browser before using it in the script.

---

## ⚠️ Warnings & Notes

* Ensure good lighting and contrast on the road for better lane detection.
* This system is for research/demo purposes and **not** production-ready for real driving environments.
* Alarm may not play if `pygame` has audio device compatibility issues—ensure proper setup.

---

## 👨‍💻 Author

**Elias Mang'era Mwita**
Undergraduate Student | Mbeya University of Science and Technology
Project Title: *Development of Wrong Way Driving Monitoring System*

---

## 📄 License

This project is open-source and free to use for educational and research purposes.

MIT License

Copyright (c) 2025 Elias Mang'era Mwita

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal 
in the Software without restriction, including without limitation the rights 
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
THE SOFTWARE.
