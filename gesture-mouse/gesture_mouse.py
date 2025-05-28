import cv2
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

screen_width, screen_height = pyautogui.size()
click_cooldown = 0.5
scroll_cooldown = 1.0
last_click_time = 0
last_scroll_time = 0

def finger_up(lm, tip_id, pip_id):
    return lm[tip_id].y < lm[pip_id].y

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        index_x = int(lm[8].x * screen_width)
        index_y = int(lm[8].y * screen_height)
        pyautogui.moveTo(index_x, index_y, duration=0.01)

        dist = ((lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2) ** 0.5
        if dist < 0.03 and (time.time() - last_click_time) > click_cooldown:
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

cap.release()
cv2.destroyAllWindows()
