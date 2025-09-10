import argparse, json, os
import torch
from torch import nn
from PIL import Image, ImageOps
import numpy as np

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def load_image_to_tensor(path):
    img = Image.open(path).convert("L")  # grayscale
    img = ImageOps.invert(img) if np.array(img).mean() > 127 else img  # try to ensure white digit on black bg
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    # normalize with MNIST stats
    arr = (arr - 0.1307) / 0.3081
    tensor = torch.from_numpy(arr)[None, None, :, :]  # (1,1,28,28)
    return tensor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=str, help="path to a 28x28 png (or any image, auto-resized)")
    ap.add_argument("--model", type=str, default="model.pth")
    ap.add_argument("--classes", type=str, default="class_names.json")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"{args.model} not found. Train first: python train.py")
    if not os.path.exists(args.classes):
        raise FileNotFoundError(f"{args.classes} not found. Train first: python train.py")

    model = SmallCNN()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    x = load_image_to_tensor(args.image)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    with open(args.classes, "r") as f:
        class_names = json.load(f)

    top = probs.argmax()
    print("Prediction:", class_names[top])
    print("Probabilities:")
    for i, p in enumerate(probs):
        print(f"  {class_names[i]}: {p:.4f}")

if __name__ == "__main__":
    main()
