# Hand Gesture-Controlled Fractal Visualizer & Media Controller

This project is an interactive desktop application that combines computer vision and visual art. It allows users to **control media playback** (play/pause, skip, volume) using **hand gestures** via webcam, while a **real-time fractal animation** responds visually to these gestures. It's built with Python using `MediaPipe`, `OpenCV`, `PyAutoGUI`, `Pygame`, and `NumPy`.

## Features

- **Hand Gesture Recognition** via webcam using MediaPipe.
- **Media Controls**:
  - **Volume control** with left-hand pinch distance.
  - **Play/Pause** with open palm.
  - **Next/Previous track** with index/pinky finger gestures.
- **Colorized Visuals** that evolves based on gestures and plays in real-time using `Pygame`.
- Runs in a windowed mode and can be resized as needed.
- Utilizes smoothing and cooldown logic for accurate and user-friendly gesture input.

## Tech Stack

- **Python**
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)
- [Pygame](https://www.pygame.org/)
- NumPy, math, time

## Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/gesture-fractal-visualizer.git
cd gesture-fractal-visualizer
```

2. **Install Dependencies**

```
pip install opencv-python mediapipe pyautogui pygame numpy
```

4. **Run the app:**

```
python gesture_fractal_controller.py
```

**Gesture Mappings**

```
Left Hand Pinch: Volume Up/Down
Right Index Up:	Next Track (Ctrl + →)
Right Pinky Up:	Previous Track
Right Open Palm: Toggle Play/Pause
Left Thumb + Middle Touch:	Exit App
```
