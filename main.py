# sudo apt-get install libopencv-dev python3-opencv
# Install required libraries
# sudo apt-get install python3-opencv python3-pip
# pip3 install opencv-python-headless
# pip3 install google-generativeai
# pip3 install cvzone
# pip3 install numpy
# pip3 install streamlit
# pip3 install pillow



import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image

import streamlit as st
import time

# Page configuration
st.set_page_config(layout="wide")
st.title("Math with Gestures")

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)  # Width
cap.set(4, 720)  # Height

# Create the layout
col1, col2 = st.columns([3, 2])

# Camera view column
with col1:
    # Create a placeholder for the video feed
    frame_placeholder = st.empty()
    run = st.checkbox('Start/Stop', value=True)

# Result column
with col2:
    st.header("Answer")
    result_placeholder = st.empty()
    status_placeholder = st.empty()

# Initialize hand detector
detector = HandDetector(
    staticMode=False,
    maxHands=1,
    modelComplexity=1,
    detectionCon=0.7,
    minTrackCon=0.5
)

# Initialize Gemini AI
try:
    genai.configure(api_key="AIzaSyCoAQ8islij94Hu9bLXgZH6pGW4oYpT1X4")
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini AI: {str(e)}")
    model = None

# Initialize variables
prev_pos = None
canvas = None
last_api_call = 0
API_CALL_COOLDOWN = 2  # seconds
processing = False


def get_hand_info(img):
    """Detect hands and return finger status"""
    try:
        hands, img = detector.findHands(img, draw=False, flipType=True)
        if hands:
            hand = hands[0]
            lm_list = hand["lmList"]
            fingers = detector.fingersUp(hand)
            return fingers, lm_list
        return None
    except Exception as e:
        st.error(f"Hand detection error: {str(e)}")
        return None


def draw(info, prev_pos, canvas, img):
    """Handle drawing operations"""
    if canvas is None:
        canvas = np.zeros_like(img)

    fingers, lm_list = info
    current_pos = None

    # Drawing mode (index finger)
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = tuple(lm_list[8][0:2])
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (0, 0, 255), 10)

    # Clear canvas (thumb)
    elif fingers == [1, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
        status_placeholder.info("Canvas cleared")

    return current_pos, canvas


def send_to_ai(model, canvas, fingers):
    """Process image with Gemini AI"""
    global last_api_call, processing

    try:
        if not model:
            status_placeholder.warning("AI model not initialized")
            return None

        if fingers == [1, 1, 1, 0, 0] and not processing:
            current_time = time.time()
            if current_time - last_api_call < API_CALL_COOLDOWN:
                status_placeholder.info("Please wait before sending another request")
                return None

            processing = True
            status_placeholder.info("Processing math problem...")

            try:
                pil_image = Image.fromarray(canvas)
                response = model.generate_content(["Solve this math problem", pil_image])
                last_api_call = current_time
                status_placeholder.success("Problem processed successfully")
                return response.text
            except Exception as api_error:
                status_placeholder.error(f"API error: {str(api_error)}")
                return None
            finally:
                processing = False

    except Exception as e:
        status_placeholder.error(f"Processing error: {str(e)}")
        processing = False
        return None


def main():
    global prev_pos, canvas

    if not cap.isOpened():
        st.error("Could not open camera")
        return

    try:
        while run:
            # Read frame from camera
            ret, img = cap.read()
            if not ret:
                st.error("Failed to get frame from camera")
                break

            # Flip the frame horizontally
            img = cv2.flip(img, 1)

            # Initialize canvas if needed
            if canvas is None:
                canvas = np.zeros_like(img)

            # Get hand information
            hand_info = get_hand_info(img)
            if hand_info:
                fingers, _ = hand_info

                # Handle drawing
                prev_pos, canvas = draw(hand_info, prev_pos, canvas, img)

                # Process with AI
                if not processing:
                    result = send_to_ai(model, canvas, fingers)
                    if result:
                        result_placeholder.markdown(f"### Result:\n{result}")

            # Combine the original image with the canvas
            combined_img = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

            # Display the combined image
            frame_placeholder.image(combined_img, channels="BGR", use_column_width=True)

            # Add a small delay to reduce CPU usage
            time.sleep(0.01)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up
        cap.release()


if __name__ == "__main__":
    main()
