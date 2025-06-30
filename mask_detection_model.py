import os
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Constants
IMG_SIZE = 128
DATA_DIR = r"C:\Users\Acer\Downloads\Python\Github\Mask Detection\Training"
CATEGORIES = ["with_mask", "without_mask"]

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("📁 Loading and processing images...")
data, labels = [], []

def extract_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces:
        return image[y:y+h, x:x+w]
    return None

for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)
    count = 0
    for img_name in os.listdir(path):
        try:
            img = cv2.imread(os.path.join(path, img_name))
            face = extract_face(img)
            if face is not None:
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                data.append(face)
                labels.append(category)
                count += 1
        except:
            pass
    print(f"📂 {category}: {count} faces")

print(f"✅ Total faces: {len(data)}")

# Encode and balance
X = np.array(data) / 255.0
le = LabelEncoder()
y = le.fit_transform(labels)
X_flat = X.reshape(len(X), -1)
X_res, y_res = RandomOverSampler().fit_resample(X_flat, y)
X = X_res.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = to_categorical(y_res)
print(f"⚖️ Balanced dataset: {X.shape[0]} samples")

# Stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, np.argmax(y, axis=1)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# Augment both classes equally
augmenter = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode='nearest'
)
augmenter.fit(X_train)

# Model building
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = BatchNormalization()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_mask_model.keras", save_best_only=True)

# Training
print("🚀 Training model...")
model.fit(
    augmenter.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=30,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# Final evaluation
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc:.4f}")
