import argparse, json, os, torch
from torch import nn

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="model.pth", help="trained PyTorch weights")
    ap.add_argument("--out", type=str, default="model.onnx", help="ONNX output path")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"{args.model} not found. Train first: python train.py")

    model = SmallCNN()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28, dtype=torch.float32)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}}
    torch.onnx.export(
        model, dummy, args.out, input_names=["input"], output_names=["logits"],
        dynamic_axes=dynamic_axes, opset_version=args.opset
    )
    print(f"Exported ONNX to {args.out}")

if __name__ == "__main__":
    main()
