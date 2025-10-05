# predict_app.py
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io
from gradcam import GradCAM, preprocess_image
import numpy as np

@st.cache_resource
def load_model(checkpoint):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device)
    classes = ckpt.get('classes', ['NORMAL','PNEUMONIA'])
    model = models.resnet18(pretrained=False)
    in_f = model.fc.in_features
    model.fc = torch.nn.Linear(in_f, len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, classes, device

st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image (JPEG / PNG). This is an academic demo only.")

checkpoint_path = st.text_input("Path to model checkpoint", "best_model.pth")
if not checkpoint_path:
    st.warning("Provide checkpoint path.")
    st.stop()

model, classes, device = load_model(checkpoint_path)

uploaded_file = st.file_uploader("Choose an X-ray image", type=["png","jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
    st.write(f"Prediction: **{classes[pred_idx]}** (confidence {conf:.3f})")
    # Grad-CAM
    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(input_tensor)
    gradcam.remove_hooks()
    heatmap = (np.uint8(255 * cam))
    import cv2
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = 0.4 * heatmap + 0.6 * (np.array(image.resize((224,224))) / 255.0)
    st.image(overlay, caption='Grad-CAM overlay', use_column_width=True)
