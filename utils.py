# utils.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_confusion_matrix(y_true, y_pred, labels=["NORMAL","PNEUMONIA"], figsize=(6,5), savepath=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_probs)
    report = classification_report(y_true, y_pred, target_names=["NORMAL","PNEUMONIA"], output_dict=True)
    return {'auc': auc, 'report': report, 'y_pred': y_pred}

def imshow_tensor(img_tensor, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
    # img_tensor: C,H,W
    img = img_tensor.cpu().numpy().transpose(1,2,0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
