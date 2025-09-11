import gradio as gr
import torch
from torch import nn
from PIL import Image, ImageOps
import numpy as np
import json, os

MODEL_PATH = "model.pth"
CLASSES_PATH = "class_names.json"

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

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Please run: python train.py")
    model = SmallCNN()
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def load_classes():
    if os.path.exists(CLASSES_PATH):
        with open(CLASSES_PATH, "r") as f:
            return json.load(f)
    return [str(i) for i in range(10)]

model = load_model()
class_names = load_classes()

def preprocess(img: Image.Image):
    img = img.convert("L")
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - 0.1307) / 0.3081
    x = torch.from_numpy(arr)[None, None, :, :]
    return x, img

@torch.no_grad()
def predict(img: Image.Image):
    if img is None:
        return None, {}
    x, preview = preprocess(img)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).numpy().tolist()
    conf = {cls: float(p) for cls, p in zip(class_names, probs)}
    return preview, conf

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上傳圖片（任意大小；自動轉 28×28 灰階）"),
    outputs=[
        gr.Image(type="pil", label="預處理後 28×28 預覽"),
        gr.Label(num_top_classes=10, label="預測結果（含機率）"),
    ],
    title="MNIST 手寫數字辨識（PyTorch + Gradio）",
    description="上傳手寫數字圖片，系統會自動轉成 28×28 灰階並預測 0–9。"
)

if __name__ == "__main__":
    demo.launch()
