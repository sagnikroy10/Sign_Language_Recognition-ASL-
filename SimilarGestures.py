import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

model = tf.keras.models.load_model(r'C:\Users\sagni\OneDrive\Desktop\DL Project\Model\keras_model.h5')

# Gesture class labels
class_names = ['A', 'D', 'E', 'M', 'N','V', 'W', '1', '2', '6']

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Gesture detection logic
def get_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    index_pip = hand_landmarks.landmark[6]
    index_mcp = hand_landmarks.landmark[5]

    middle_tip = hand_landmarks.landmark[12]
    middle_pip = hand_landmarks.landmark[10]

    ring_tip = hand_landmarks.landmark[16]
    ring_pip = hand_landmarks.landmark[14]

    pinky_tip = hand_landmarks.landmark[20]
    pinky_pip = hand_landmarks.landmark[18]

    ring_mcp = hand_landmarks.landmark[13]
    ring_tip = hand_landmarks.landmark[16]

    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    ring_extended = ring_tip.y < hand_landmarks.landmark[13].y
    pinky_extended = pinky_tip.y < hand_landmarks.landmark[18].y

    thumb_tip = hand_landmarks.landmark[4]
    ring_knuckle = hand_landmarks.landmark[13]  
    pinky_knuckle = hand_landmarks.landmark[17] 
    pinky_knuckle = hand_landmarks.landmark[17] 
    pinky_nail = hand_landmarks.landmark[20]  

    # Finger status
    index_curled = index_tip.y > index_pip.y
    middle_curled = middle_tip.y > middle_pip.y
    ring_curled = ring_tip.y > ring_pip.y
    pinky_curled = pinky_tip.y > pinky_pip.y

    index_extended = index_tip.y < index_mcp.y
    middle_extended = middle_tip.y < middle_pip.y
    pinky_curled = pinky_tip.y > hand_landmarks.landmark[18].y  

    # Distances
    dist_thumb_index_mcp = distance(thumb_tip, index_mcp)
    dist_thumb_middle_tip = distance(thumb_tip, middle_tip)
    dist_thumb_ring_mcp = distance(thumb_tip, ring_mcp)
    dist_thumb_ring_tip = distance(thumb_tip, ring_tip)
    dist_thumb_pinky_knuckle = distance(thumb_tip, pinky_knuckle)
    dist_thumb_pinky_nail = distance(thumb_tip, pinky_nail)


    threshold_ratio = 0.9  

    # Check for M gesture (thumb at pinky knuckle)
    threshold = 0.05 


    # For M or N gestures
    if abs(thumb_tip.x - pinky_knuckle.x) < threshold and abs(thumb_tip.y - pinky_knuckle.y) < threshold:
        if hand_landmarks.landmark[5].y > thumb_tip.y and hand_landmarks.landmark[9].y > thumb_tip.y and hand_landmarks.landmark[13].y > thumb_tip.y:
            return "M"  # M gesture: Thumb touching pinky knuckle

    if abs(thumb_tip.x - ring_knuckle.x) < threshold and abs(thumb_tip.y - ring_knuckle.y) < threshold:
        if hand_landmarks.landmark[5].y > thumb_tip.y and hand_landmarks.landmark[9].y > thumb_tip.y and hand_landmarks.landmark[13].y > thumb_tip.y:
            return "N"  # N gesture: Thumb touching ring knuckle
    
    # For A or E gestures
    if index_curled and middle_curled and ring_curled and pinky_curled:
        if thumb_tip.x > index_tip.x:
            return "A"
        else:
            return "E"   
        
    # For W or 6 gestures
    if index_extended and middle_extended and ring_extended and pinky_curled:
        if abs(dist_thumb_pinky_knuckle) < threshold:
            return "W"  # W gesture: Thumb touching pinky knuckle

    if index_extended and middle_extended and ring_extended and pinky_curled:
        if abs(dist_thumb_pinky_nail) < threshold:
            return "6"  # 6 gesture: Thumb touching pinky nail

    # For 1 or D  & V or 2 gestures
    if ring_curled and pinky_curled:
            # Case 1: Only index is extended
            if index_extended and not middle_extended:
                if dist_thumb_index_mcp < dist_thumb_middle_tip * threshold_ratio:
                    return "1"
                elif dist_thumb_middle_tip < dist_thumb_index_mcp * threshold_ratio:
                    return "D"

            # Case 2: Both index and middle extended
            if index_extended and middle_extended:
                if dist_thumb_ring_mcp < dist_thumb_ring_tip * threshold_ratio:
                    return "2"
                elif dist_thumb_ring_tip < dist_thumb_ring_mcp * threshold_ratio:
                    return "V"

    return None


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            gesture = get_gesture(landmarks)
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            if gesture:
                cv2.putText(frame, f'Gesture: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
