import torch
from pathlib import Path
from torchvision.io import read_video

from model import build_model
from transforms import get_video_transform

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

def sample_frames(video, num_frames=16):
    T = video.shape[0]
    if T <= num_frames:
        idxs = list(range(T))
        while len(idxs) < num_frames:
            idxs += idxs
        return idxs[:num_frames]
    else:
        step = T / num_frames
        return [int(i * step) for i in range(num_frames)]

def load_video_tensor(video_path, num_frames=16):
    video, _, _ = read_video(str(video_path), pts_unit="sec")  # (T, H, W, C)
    idxs = sample_frames(video, num_frames)
    video = video[idxs]  # (num_frames, H, W, C)

    video = video.float() / 255.0
    video = video.permute(0, 3, 1, 2)  # (T, C, H, W)

    transform = get_video_transform()
    video = torch.stack([transform(frame) for frame in video], dim=0)

    # Add batch dimension & reorder to (B, C, T, H, W)
    video = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return video

def predict(video_path, checkpoint="video_classifier.pt"):
    print(f"Loading model from: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device)

    class_names = ckpt["class_names"]

    model = build_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    clip = load_video_tensor(video_path).to(device)

    with torch.no_grad():
        logits = model(clip)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, 1)

    predicted_class = class_names[idx.item()]
    return predicted_class, conf.item()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/video.mp4")
        exit(1)

    video_path = Path(sys.argv[1])
    pred, confidence = predict(video_path)
    print(f"\nPrediction: {pred}\nConfidence: {confidence:.3f}")

