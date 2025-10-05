# gradcam.py
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import streamlit as st
import cv2

# -----------------------------
# GradCAM Implementation
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        handle_f = self.target_layer.register_forward_hook(forward_hook)
        handle_b = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.extend([handle_f, handle_b])

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1).item()
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients[0]      # C x H x W
        acts = self.activations[0]     # C x H x W
        weights = torch.mean(grads, dim=(1, 2))
        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return cam


# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(img_path, img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    tensor = tf(img).unsqueeze(0)
    return tensor, np.array(img)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get('classes', ['NORMAL', 'PNEUMONIA'])

    # Load model
    model = models.resnet18(pretrained=False)
    in_f = model.fc.in_features
    model.fc = torch.nn.Linear(in_f, len(classes))
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # Grad-CAM setup
    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)

    input_tensor, orig_img = preprocess_image(args.image, img_size=args.img_size)
    input_tensor = input_tensor.to(device)
    cam = gradcam(input_tensor)
    gradcam.remove_hooks()

    # -----------------------------
    # Grad-CAM Visualization
    # -----------------------------
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))

    # Normalize original image
    orig_img = orig_img.astype(np.float32) / 255.0

    # Overlay
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
    overlay = np.clip(overlay, 0, 1)

    # Display in Streamlit
    st.image(orig_img, caption='Original Image', use_column_width=True, clamp=True)
    st.image(overlay, caption='Grad-CAM Overlay', use_column_width=True, clamp=True)
