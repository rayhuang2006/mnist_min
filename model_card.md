# Model Card: MNIST-Min SmallCNN

## Overview
- **Model**: SmallCNN (2 conv layers + 2x2 maxpool + 2 FC)
- **Task**: Handwritten digit classification (0–9)
- **Framework**: PyTorch (exportable to ONNX / TFLite)
- **Intended Use**: Educational / demo; not for production without further validation

## Data
- **Dataset**: MNIST (train=60k, test=10k), grayscale 28×28
- **Source**: torchvision.datasets.MNIST
- **Preprocessing**: ToTensor + Normalize(mean=0.1307, std=0.3081)

## Training
- **Loss**: CrossEntropy
- **Optimizer**: Adam (lr=1e-3 by default)
- **Batch Size**: 128 (default)
- **Epochs**: 2（demo；建議 3–5 以獲得更高準確率）
- **Hardware**: CPU or CUDA GPU

## Metrics (typical)
- **Accuracy**: ~98% on test after 2–3 epochs

## Limitations
- Expects 28×28 grayscale inputs, digit foreground ~white on dark background.
- Not robust to rotations or non-MNIST-style handwriting.

## Ethical Considerations
- Educational dataset；no PII.
- For real-world use, consider fairness, bias, and misuse risks.

## Export / Deployment
- **ONNX**: `python export_onnx.py --model model.pth --out model.onnx`
- **TFLite**: `python export_tflite.py --onnx model.onnx --out model.tflite [--quant int8]`
- **ONNX Runtime**: `python onnxruntime_infer.py --onnx model.onnx --image samples/0.png`
