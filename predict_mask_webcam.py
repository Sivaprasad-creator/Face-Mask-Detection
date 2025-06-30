import cv2
import numpy as np
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

# Load the model
model = load_model("best_mask_model.keras")

IMG_SIZE = 128
LABELS = ["with_mask", "without_mask"]
LABEL_MAP = {
    "with_mask": "Mask Detected",
    "without_mask": "No Mask Detected"
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def preprocess_face(face):
    resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 3)

# Open webcam
print("📷 Starting webcam... Press 'q' to quit.")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None or frame.size == 0:
        print("⚠️ Skipping empty frame...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]

        if face_roi.size > 0:
            input_img = preprocess_face(face_roi)
            pred = model.predict(input_img, verbose=0)
            pred_label = LABELS[np.argmax(pred)]
            prediction = LABEL_MAP[pred_label]
            confidence = np.max(pred)

            color = (0, 255, 0) if pred_label == "with_mask" else (0, 0, 255)
            label_text = f"{prediction} ({confidence * 100:.1f}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
