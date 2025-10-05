Pneumonia Detection from Chest X-Rays using Deep Learning

This project implements a deep learning-based system for detecting pneumonia from chest X-ray images using a ResNet-18 model and visualizes model predictions using Grad-CAM. The project also includes evaluation metrics and a Streamlit-based interface for interactive visualization.

Project Overview
Pneumonia is a serious lung infection, and early diagnosis is critical. This project uses Convolutional Neural Networks (CNNs) to automatically classify chest X-ray images as NORMAL or PNEUMONIA.

Key Components:

Pretrained ResNet-18 modified for binary classification.
Grad-CAM visualization to highlight areas contributing to model decisions.
Streamlit interface for interactive image upload and prediction.
Evaluation metrics including accuracy, precision, recall, F1-score, and AUC.

Features

Predict pneumonia from a single chest X-ray image.
Display Grad-CAM heatmap overlay on the original X-ray.
Evaluate model on a test dataset with detailed metrics.
Easy-to-use web interface via Streamlit.

Dataset

This project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle
Training set: Normal + Pneumonia images
Validation set: Normal + Pneumonia images
Test set: Normal + Pneumonia images

Installation

Clone the repository:
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

Install dependencies:
pip install -r requirements.txt

Dependencies include:
torch, torchvision
numpy, matplotlib, Pillow
opencv-python
streamlit

Usage
1. Evaluate Model
python evaluate.py --checkpoint best_model.pth --test_dir chest_xray/test --batch_size 16 --img_size 224
Outputs accuracy, precision, recall, F1-score, and AUC.
2. Generate Grad-CAM
streamlit run gradcam.py
Upload an X-ray image to view prediction and Grad-CAM overlay.
3. Predict from Script
python gradcam.py --checkpoint best_model.pth --image path/to/image.jpeg --img_size 224
Generates Grad-CAM overlay and displays prediction.
Model Architecture
Base Model: ResNet-18
Modification: Fully connected layer replaced to output 2 classes (NORMAL, PNEUMONIA)
Loss Function: Cross-Entropy Loss
Optimizer: Adam / SGD (as per training)
Input Size: 224x224 RGB images

Evaluation

Test Accuracy: ~87.7

AUC: 0.9646

F1-Score:
NORMAL: 0.807

PNEUMONIA: 0.909

Observation: High recall for pneumonia detection; slightly lower for NORMAL.

Results
Grad-CAM Visualization: Highlights lung regions contributing to pneumonia detection.
Streamlit Demo: Users can upload X-rays and see real-time prediction with heatmap overlay.

Future Work
Improve NORMAL class recall via data augmentation.
Experiment with Deeper CNNs like ResNet-50, DenseNet.
Integrate with clinical data for multi-modal diagnosis.

Deploy as a web application for hospitals or telemedicine.
