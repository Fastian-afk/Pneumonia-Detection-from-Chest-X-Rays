# 🩺 Chest X-ray Pneumonia Detection with Grad-CAM

This project visualizes attention maps using **Grad-CAM** on a ResNet18 model trained for binary classification of chest X-ray images (NORMAL vs PNEUMONIA).

The goal is to understand **where the model focuses** when diagnosing pneumonia, providing interpretability and transparency to deep learning in healthcare.


## 🔍 Features

- ✅ Pretrained ResNet18 backbone
- ✅ Grad-CAM visualizations (torchcam)
- ✅ Support for grayscale X-rays (converted to 3 channels)
- ✅ Easy-to-modify code for other datasets or layers
- ✅ Saves heatmaps as PNGs for sharing/reporting


## 📁 Folder Structure

chest-xray-gradcam/
│
├── main.py # Script to generate Grad-CAM heatmaps
├── model.py # Model loading (ResNet18 with sigmoid head)
├── transforms.py # X-ray image preprocessing
├── utils.py # CAM logic & visualization
└── gradcam_outputs/ # Saved heatmap images

## 📦 Requirements

pip install torch torchvision torchcam matplotlib pillow
🚀 Run
Make sure your validation dataset is structured like:

css
Copy
Edit
val/
├── NORMAL/
├── PNEUMONIA/
Then run:

bash
Copy
Edit
python main.py
All heatmaps will be saved to gradcam_outputs/.

🤖 Credits:
Created by Imaad Fazal — Knowledge Discovery & Data Science Lab
Roll No: 23I-0656

📜 License:
This project is open source under the MIT License.
