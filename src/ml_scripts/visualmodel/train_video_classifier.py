# train_video_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataloaders import get_dataloaders
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")
    for videos, labels in pbar:
        # videos: (B, T, C, H, W) -> (B, C, T, H, W)
        videos = videos.permute(0, 2, 1, 3, 4).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    return running_loss / total, 100. * correct / total

def eval_model(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.permute(0, 2, 1, 3, 4).to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    return running_loss / total, 100. * correct / total

def main():
    train_loader, test_loader, classes = get_dataloaders(root="../../../Data/Visual_split/", batch_size=1, num_frames=16)
    num_classes = len(classes)

    model = build_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = eval_model(model, test_loader, criterion)

        print(f"Epoch {epoch}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": classes,
    }, "video_classifier.pt")

if __name__ == "__main__":
    main()

