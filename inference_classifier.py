import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./modelV2.p', 'rb'))
model = model_dict['model']

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

# Placeholder character for no specific sign detected
placeholder_character = '?'
confidence_threshold = 0.5  # Threshold for displaying placeholder

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    data_aux = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Collect landmark data
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i < 21:  # Ensure that only 21 landmarks are processed
                    data_aux.extend([landmark.x, landmark.y])

            # Break after processing the first hand
            break

    # Ensure data_aux has the correct number of features (42)
    if len(data_aux) == 42:
        data_aux = np.asarray(data_aux).reshape(1, -1)
        prediction = model.predict(data_aux)
        predicted_character = prediction[0]  # Directly use the string prediction

        # Calculate confidence score
        prediction_proba = model.predict_proba(data_aux)
        confidence_score = np.max(prediction_proba)

        if confidence_score < confidence_threshold:
            predicted_character = placeholder_character

        # Display the predicted or placeholder character on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predicted_character, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
