import cv2
import mediapipe as mp
import pyautogui
import math
import time
from collections import deque
import pygame
import numpy as np

def draw_fractal(surface, zoom, offset, color_shift):
    width, height = surface.get_size()

    # Low-res for speed
    low_res = (200, 150)
    arr = np.zeros((low_res[1], low_res[0], 3), dtype=np.uint8)

    # Generate X, Y coordinate grids
    x = np.linspace(-1, 1, low_res[0]) * zoom + offset[0]
    y = np.linspace(-1, 1, low_res[1]) * zoom + offset[1]
    X, Y = np.meshgrid(x, y)

    # Trippy math: distance from center and cosine pattern
    dist = np.sqrt(X**2 + Y**2)
    colors = (np.cos(dist * 10 - color_shift * 0.1) * 127 + 128).astype(np.uint8)

    arr[..., 0] = (colors + color_shift) % 256
    arr[..., 1] = (colors * 0.5 + color_shift * 2) % 256
    arr[..., 2] = (255 - colors + color_shift * 3) % 256

    visual_surface = pygame.transform.scale(
        pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2))), (width, height)
    )
    surface.blit(visual_surface, (0, 0))

pygame.init()
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH // 2, HEIGHT // 2))
pygame.display.set_caption("Fractal Visualizer")
clock = pygame.time.Clock()

zoom = 1.0
offset = [0.0, 0.0]
paused = True
color_shift = 0
speed_factor = 0.01

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

volume_history = deque(maxlen=5)
prev_volume_send_time = time.time()
prev_volume = None

# Gesture cooldowns
last_skip_time = 0.1
last_pause_time = 0.1
gesture_cooldown = 1.0 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)


    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            lm = hand_landmarks.landmark
            coords = [(int(p.x * w), int(p.y * h)) for p in lm]

            if label == 'Left':
                thumb, index, middle = coords[4], coords[8], coords[12]
                stop = distance(thumb, middle)
                d = distance(thumb, index)
                scaled = max(min((d - 10) / 160 * 0.05, 0.05), 0.002)
                zoom += scaled
                raw_volume = int(min(max((d - 20) / 160 * 100, 0), 100))
                volume_history.append(raw_volume)
                smooth_volume = int(sum(volume_history) / len(volume_history))

                now = time.time()
                if now - prev_volume_send_time > 0.5:
                    prev_volume_send_time = now
                    print(f"Smooth Volume: {smooth_volume}%")

                    if prev_volume is None:
                        prev_volume = smooth_volume

                    diff = smooth_volume - prev_volume
                    if abs(diff) >= 5:
                        if diff > 0:
                            pyautogui.press("volumeup", presses=int(diff / 2))
                        else:
                            pyautogui.press("volumedown", presses=int(abs(diff) / 2))
                        prev_volume = smooth_volume
                
                if stop - 20 < 0:
                    print("Middle/Thumb Connection → Exiting app")
                    cap.release()
                    cv2.destroyAllWindows()

            elif label == 'Right':
                now = time.time()

                index_tip_y = coords[8][1]
                index_base_y = coords[6][1]
                middle_tip_y = coords[12][1]
                middle_base_y = coords[10][1]
                ring_tip_y = coords[16][1]
                ring_base_y = coords[14][1]
                pinky_tip_y = coords[20][1]
                pinky_base_y = coords[18][1]

                if index_tip_y < index_base_y - 25 and now - last_skip_time > gesture_cooldown and middle_tip_y > middle_base_y - 10:
                    print("Index finger raised → Skip track")
                    color_shift += 30
                    pyautogui.hotkey("ctrl", "right")
                    last_skip_time = now
        
                if pinky_tip_y < pinky_base_y - 25 and now - last_skip_time > gesture_cooldown and middle_tip_y > middle_base_y - 10:
                    print("Pinky raised → Previous track/Restart track")
                    color_shift += 30
                    pyautogui.hotkey("ctrl", "left")
                    last_skip_time = now

                fingertips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
                bases = [6, 10, 14, 18]
                fingers_up = all(coords[tip][1] < coords[base][1] - 20 for tip, base in zip(fingertips, bases))

                thumb_tip_x = coords[4][1]
                thumb_base_x = coords[2][1]
                thumb_open = abs(thumb_tip_x - thumb_base_x) > 40

                if fingers_up and thumb_open and now - last_pause_time > gesture_cooldown:

                    paused = not paused
                    print("Open Palm → Play/Pause")
                    pyautogui.press("playpause")
                    last_pause_time = now

    cv2.imshow("Spotify Hand Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    if paused == False:
        draw_fractal(screen, zoom, offset, color_shift)
        pygame.display.flip()
        clock.tick(30)

cap.release()
cv2.destroyAllWindows()
