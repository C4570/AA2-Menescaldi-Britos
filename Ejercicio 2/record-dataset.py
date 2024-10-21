import cv2
import numpy as np
import mediapipe as mp
import os

# Create a directory "Ejercicio 2" to save the dataset if it doesn't exist
if not os.path.exists('Ejercicio 2'):
    os.makedirs('Ejercicio 2')

# Paths for saving the dataset
dataset_path = os.path.join('Ejercicio 2', 'rps_dataset.npy')
labels_path = os.path.join('Ejercicio 2', 'rps_labels.npy')

# Initialize MediaPipe Hand Landmarker
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Load or create the dataset
if os.path.exists(dataset_path) and os.path.exists(labels_path):
    dataset = np.load(dataset_path, allow_pickle=True).tolist()
    labels = np.load(labels_path, allow_pickle=True).tolist()
else:
    dataset = []
    labels = []

# Open the webcam
cap = cv2.VideoCapture(0)

# Resize the window to a larger size (800x600)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Gesture map and label instructions
gesture_map = {1: 'Piedra', 2: 'Tijeras', 3: 'Papel'}
gesture_label = {49: 0, 50: 2, 51: 1}  # 49='1' for Piedra, 50='2' for Tijeras, 51='3' for Papel

# Load emoji images with alpha channel (transparency)
rock_emoji = cv2.imread('Ejercicio 2\emojis/rock.png', cv2.IMREAD_UNCHANGED)  # Includes alpha channel
scissors_emoji = cv2.imread('Ejercicio 2\emojis/scissors.png', cv2.IMREAD_UNCHANGED)  # Includes alpha channel
paper_emoji = cv2.imread('Ejercicio 2\emojis\paper.png', cv2.IMREAD_UNCHANGED)  # Includes alpha channel

# Resize emoji images to fit on screen
emoji_size = 25
rock_emoji = cv2.resize(rock_emoji, (emoji_size, emoji_size))
scissors_emoji = cv2.resize(scissors_emoji, (emoji_size, emoji_size))
paper_emoji = cv2.resize(paper_emoji, (emoji_size, emoji_size))

def overlay_png(frame, emoji, x, y):
    """ Overlay a transparent PNG onto the video frame at position (x, y). """
    # Get the dimensions of the emoji
    h, w, _ = emoji.shape
    
    # Extract the RGB and alpha channels of the emoji
    emoji_rgb = emoji[:, :, :3]  # The first 3 channels are RGB
    alpha_mask = emoji[:, :, 3]  # The 4th channel is the alpha mask
    
    # Get the region of interest (ROI) in the frame where we will place the emoji
    roi = frame[y:y + h, x:x + w]
    
    # Create an inverse mask to use on the ROI (the non-transparent part of the emoji)
    alpha_inv = cv2.bitwise_not(alpha_mask)
    
    # Black-out the area of the emoji in the ROI
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (alpha_inv // 255)
    
    # Add the emoji to the ROI using the alpha mask
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] + (emoji_rgb[:, :, c] * (alpha_mask // 255))

def draw_instructions(frame):
    """ Draw the instructions for the user on the video frame using emoji images. """
    font = cv2.FONT_HERSHEY_SIMPLEX
    overlay = frame.copy()

    # Create a semi-transparent background for the text
    cv2.rectangle(overlay, (0, 0), (210, 190), (0, 0, 0), -1)  # Black background rectangle
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)  # Blend with the original frame

    # Add text for the instructions with new positions
    cv2.putText(frame, "Presiona:", (10, 30), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "1 para", (80, 70), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "2 para", (80, 100), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "3 para", (80, 130), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "'Q' para salir", (10, 170), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Overlay the emoji images next to the text
    overlay_png(frame, rock_emoji, 170, 50)  # 1 para 🪨
    overlay_png(frame, scissors_emoji, 170, 80)  # 2 para ✂️
    overlay_png(frame, paper_emoji, 170, 110)  # 3 para 📄
    
def record_gesture(gesture):
    global dataset, labels

    success, frame = cap.read()
    if not success:
        return

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hand landmarks
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])

            # Append landmarks and label to the dataset
            dataset.append(np.array(landmarks).flatten())
            labels.append(gesture)

            # Visualize landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame with instructions
    draw_instructions(frame)
    cv2.imshow('Recording Gesture', frame)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Draw instructions on the video
    draw_instructions(frame)
    
    # Show the frame
    cv2.imshow('Recording Gesture', frame)

    # Wait for keypress
    key = cv2.waitKey(1)

    # If 'q' is pressed, break the loop and exit
    if key == ord('q'):
        break

    # Record gestures based on keypress
    if key in gesture_label:
        record_gesture(gesture_label[key])

# Save the dataset to the "Ejercicio 2" folder
np.save(dataset_path, np.array(dataset))
np.save(labels_path, np.array(labels))

# Cleanup
cap.release()
cv2.destroyAllWindows()
