

# # Install OpenCV (optimized for headless systems)
# pip3 install opencv-python-headless

# # Install PyTorch (compatible with Jetson Nano)
# pip3 install torch torchvision

# # Install YOLOv8 and the Ultralytics library
# pip3 install ultralytics

# # Install Pillow for image handling
# pip3 install Pillow

# # Install Hugging Face Transformers for classification
# pip3 install transformers

# # Install TeleBot for Telegram bot integration
# pip3 install telebot

# # Install Requests for API handling (e.g., Blynk, Telegram)
# pip3 install requests


# sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libjpeg-dev zlib1g-dev
# pip3 install numpy torch==1.10.0 torchvision==0.11.0









import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import time
import requests
import telebot
from ultralytics import YOLO

# Initialize the YOLO model (optimized for Jetson Nano; TensorRT recommended for further optimization)
yolo_model = YOLO('yolov8n.pt')  # Load YOLO nano model (ensure the file exists in the working directory)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
yolo_model.to(device)

# Load a specialized model for precise classification
classifier_name = "microsoft/resnet-50"
feature_extractor = AutoFeatureExtractor.from_pretrained(classifier_name)
classifier_model = AutoModelForImageClassification.from_pretrained(classifier_name).to(device)

# Initialize the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set resolution to 640x480 for optimal performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Image transformation for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Animal categories mapping
animal_categories = {
    'elephant': 'Elephant', 'african_elephant': 'Elephant', 'indian_elephant': 'Elephant', 'loxodonta_africana': 'Elephant',
    'peacock': 'Peacock',
    'pig': 'Pig', 'wild_boar': 'Pig', 'domestic_pig': 'Pig', 'sus_scrofa': 'Pig',
    'wire-haired_fox_terrier': 'Wire-haired Fox Terrier', 'fox_terrier': 'Wire-haired Fox Terrier',
    'macaque': 'Macaque',
}

current_category = ""

# Blynk API token and URLs
blynk_url_on = "https://blynk.cloud/external/api/update?token=Iom3jPBDZK0SSrML2osPq3047m3u&v12=1"
blynk_url_off = "https://blynk.cloud/external/api/update?token=Iom3jPBDZK0SSrML2osPq3047m3u&v12=0"

# Telegram bot settings
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Function to send a Telegram message
def send_telegram_message(message):
    try:
        bot.send_message(TELEGRAM_CHAT_ID, message)
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

# Function to draw text with a bold border
def draw_bold_text(img, text, pos, font, font_scale, text_color, font_thickness):
    border_color = (0, 0, 0)
    border_thickness = font_thickness + 2
    cv2.putText(img, text, pos, font, font_scale, border_color, border_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

# Create a named window for full-screen display
cv2.namedWindow('Animal Detection', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Animal Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

recently_detected = set()  # Track recently detected animals

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Camera frame not available")
        break

    # Resize frame to 640x480 for consistency
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection using YOLO
    results = yolo_model(frame)
    detected_target_animal = False

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            # Focus on specific animals
            if class_name in ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'monkey']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                animal_image = frame[y1:y2, x1:x2]
                pil_image = Image.fromarray(cv2.cvtColor(animal_image, cv2.COLOR_BGR2RGB))
                inputs = feature_extractor(images=pil_image, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = classifier_model(**inputs)

                predicted_class_idx = outputs.logits.argmax(-1).item()
                predicted_class = classifier_model.config.id2label[predicted_class_idx].lower()
                confidence = torch.nn.functional.softmax(outputs.logits, dim=-1)[0, predicted_class_idx].item()

                # Update current category
                current_category = animal_categories.get(predicted_class, predicted_class)

                print(f"Detected: {predicted_class}, Categorized as: {current_category}")

                # Check if it's a target animal
                if current_category in ['Elephant', 'Peacock', 'Pig', 'Wire-haired Fox Terrier'] or \
                   any(animal in predicted_class for animal in ['elephant', 'peacock', 'pig', 'boar', 'fox_terrier']):
                    detected_target_animal = True
                    if current_category not in recently_detected:
                        print(f"{current_category} detected, activating LED and sending Telegram message")
                        requests.get(blynk_url_on)
                        send_telegram_message(f"Detected: {current_category}")
                        time.sleep(3)
                        requests.get(blynk_url_off)
                        recently_detected.add(current_category)

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f'{predicted_class}: {confidence:.2f}'
                draw_bold_text(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Clear recently_detected set if no target animal is detected
    if not detected_target_animal:
        recently_detected.clear()

    # Display the current animal category
    draw_bold_text(frame, f"Category: {current_category}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imshow('Animal Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
