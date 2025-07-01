# 😷 Face Mask Detection (Real-Time)

![GitHub stars](https://img.shields.io/github/stars/Sivaprasad-creator/Face-Mask-Detection)
![GitHub forks](https://img.shields.io/github/forks/Sivaprasad-creator/Face-Mask-Detection)
![GitHub license](https://img.shields.io/github/license/Sivaprasad-creator/Face-Mask-Detection)

---

https://github.com/Sivaprasad-creator/Face-Mask-Detection/blob/a873fc61259bed6752314a8c9500071295e937b1/Face%20Mask.png

## 📌 Project Overview

This project implements a **real-time face mask detection system** using deep learning and computer vision. It detects faces from webcam feed using OpenCV and classifies whether a person is wearing a mask or not using a fine-tuned **MobileNetV2 CNN model**.

> 🔧 **Tools Used:** Python, TensorFlow, OpenCV, Keras, imbalanced-learn, MobileNetV2

---

## 📁 Dataset Information

- **📦 Source:** Kaggle  
- **🗓️ Records:** 15,310 images  
- **📂 Classes:**
  - `with_mask`
  - `without_mask`
- **📸 Images:** Collected from real-world photos of people with and without face masks  
- **🧾 Format:** `.jpg` and `.png` images across two folders

---

## 🎯 Objectives

- Detect human faces in real time using webcam  
- Identify whether the detected face is wearing a mask  
- Solve class imbalance using oversampling  
- Improve model accuracy using transfer learning (MobileNetV2)  
- Display live predictions with bounding boxes and confidence scores

---

## 📊 Analysis Summary

- Used OpenCV Haar cascade to detect faces  
- Balanced dataset using RandomOverSampler (due to under-representation of `with_mask`)  
- Applied data augmentation to improve generalization  
- Fine-tuned MobileNetV2’s last 50 layers for transfer learning  
- Achieved better performance than traditional CNNs with minimal overfitting  
- Real-time inference with accurate visual feedback on webcam

---

## 🧠 Models Used

| Category           | Techniques                                             |
|--------------------|--------------------------------------------------------|
| **Face Detection** | Haar Cascade Classifier (OpenCV)                       |
| **Classifier**     | MobileNetV2 + Custom Dense Layers                      |
| **Augmentation**   | ImageDataGenerator (rotation, shift, brightness, etc.) |
| **Balancing**      | RandomOverSampler from `imbalanced-learn`             |
| **Metrics**        | Accuracy, Confidence (Softmax Output)                 |

---

## 🚀 How to Run Locally

1. **Download the Dataset**  
   Create the following directory structure:

   ```
   C:/Users/Acer/Downloads/Python/Github/Mask Detection/Training/
   ├── with_mask/
   └── without_mask/
   ```

2. **Run the Training Script**
   ```bash
   python mask_detection_model.py
   ```

3. **Start the Webcam Prediction**
   ```bash
   python predict_mask_webcam.py
   ```

4. ✅ Press `q` to quit webcam preview

---

## 📂 Project Structure

```
📁 Mask Detection
├── mask_detection_model.py        # Model training and saving
├── predict_mask_webcam.py         # Real-time detection via webcam
├── /Training
│   ├── with_mask/
│   └── without_mask/
└── best_mask_model.keras          # Saved MobileNetV2 model
```

---

## 📬 Author Info

**Sivaprasad T.R**  
📧 Email: [sivaprasadtrwork@gmail.com](mailto:sivaprasadtrwork@gmail.com)  
🔗 LinkedIn: [linkedin.com/in/sivaprasad-t-r](https://www.linkedin.com/in/sivaprasad-t-r)  
💻 GitHub: [github.com/Sivaprasad-creator](https://github.com/Sivaprasad-creator)

---

## 📜 Acknowledgements

- Kaggle for dataset  
- TensorFlow and Keras teams for deep learning APIs  
- OpenCV community for robust face detection  
- imbalanced-learn for oversampling tools

---

## 💬 Feedback

Have suggestions or ideas?  
Raise an issue or reach out via LinkedIn or email!
