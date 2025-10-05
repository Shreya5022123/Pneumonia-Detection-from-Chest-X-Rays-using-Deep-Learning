# train.py
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import set_seed, compute_metrics, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import json

def get_dataloaders(data_dir, batch_size=32, img_size=224, val_split=False):
    # transforms
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf) if os.path.isdir(val_dir) else None
    test_ds = datasets.ImageFolder(test_dir, transform=val_tf) if os.path.isdir(test_dir) else None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True) if test_ds else None

    class_names = train_ds.classes
    return train_loader, val_loader, test_loader, class_names

def build_model(num_classes=2, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    preds = []
    probs = []
    targets = []
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        soft = torch.softmax(outputs, dim=1)[:,1].detach().cpu().numpy()
        probs.extend(list(soft))
        preds.extend(list(torch.argmax(outputs, dim=1).cpu().numpy()))
        targets.extend(list(labels.cpu().numpy()))
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(preds), np.array(probs), np.array(targets)

@torch.no_grad()
def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    probs = []
    targets = []
    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        soft = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
        probs.extend(list(soft))
        preds.extend(list(torch.argmax(outputs, dim=1).cpu().numpy()))
        targets.extend(list(labels.cpu().numpy()))
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(preds), np.array(probs), np.array(targets)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = get_dataloaders(args.data_dir, args.batch_size, args.img_size)
    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = {'train_loss':[], 'val_loss':[], 'train_auc':[], 'val_auc':[]}

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_preds, train_probs, train_targets = train_epoch(model, train_loader, criterion, optimizer, device)
        train_auc = roc_auc_score(train_targets, train_probs)
        val_loss, val_preds, val_probs, val_targets = (None,None,None,None)
        if val_loader:
            val_loss, val_preds, val_probs, val_targets = eval_model(model, val_loader, criterion, device)
            val_auc = roc_auc_score(val_targets, val_probs)
            scheduler.step(val_loss)
        else:
            val_loss = 0
            val_auc = 0

        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_auc'].append(float(train_auc)); history['val_auc'].append(float(val_auc))

        print(f"Train Loss: {train_loss:.4f} AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} AUC: {val_auc:.4f}")

        # save best
        if val_loader and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'classes': classes}, args.checkpoint)
            print("Saved best model.")

    # final test evaluation
    if test_loader:
        print("Evaluating on test set...")
        _, test_preds, test_probs, test_targets = eval_model(model, test_loader, criterion, device)
        import numpy as np
        from utils import plot_confusion_matrix
        metrics = compute_metrics(test_targets, test_probs)
        print("Test AUC:", metrics['auc'])
        print("Classification Report:")
        import json
        print(json.dumps(metrics['report'], indent=2))
        plot_confusion_matrix(test_targets, metrics['y_pred'], labels=classes)
        # save history
        with open('history.json', 'w') as f:
            json.dump(history, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='chest_xray', help='path to data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', default='best_model.pth')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
