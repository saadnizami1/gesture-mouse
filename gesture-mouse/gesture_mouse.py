import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.PAUSE = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Could not access the webcam. Make sure it isn't being used by another app.")

screen_width, screen_height = pyautogui.size()

FRAME_MARGIN = 0.15      # shrinks the tracked region so screen corners are reachable without stretching
SMOOTHING_FACTOR = 0.4   # 0-1, how much of the distance to the target is covered each frame; lower = smoother but laggier
CLICK_DISTANCE = 0.03
click_cooldown = 0.5
scroll_cooldown = 1.0
last_click_time = 0
last_scroll_time = 0
cursor_x, cursor_y = screen_width / 2, screen_height / 2


def finger_up(lm, tip_id, pip_id):
    return lm[tip_id].y < lm[pip_id].y


def map_to_screen(x, y):
    x = (x - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)
    y = (y - FRAME_MARGIN) / (1 - 2 * FRAME_MARGIN)
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    return x * screen_width, y * screen_height


try:
    while True:
        success, frame = cap.read()
        if not success:
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark

            target_x, target_y = map_to_screen(lm[8].x, lm[8].y)
            cursor_x += (target_x - cursor_x) * SMOOTHING_FACTOR
            cursor_y += (target_y - cursor_y) * SMOOTHING_FACTOR
            pyautogui.moveTo(int(cursor_x), int(cursor_y))

            dist = ((lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2) ** 0.5
            if dist < CLICK_DISTANCE and (time.time() - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = time.time()

            index_up = finger_up(lm, 8, 6)
            middle_up = finger_up(lm, 12, 10)
            ring_up = finger_up(lm, 16, 14)
            pinky_up = finger_up(lm, 20, 18)

            if index_up and middle_up and not ring_up and not pinky_up:
                if time.time() - last_scroll_time > scroll_cooldown:
                    pyautogui.scroll(-300)
                    last_scroll_time = time.time()

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Mouse", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
