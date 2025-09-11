import argparse, json, os, numpy as np
from PIL import Image, ImageOps

def load_image_28x28(path):
    img = Image.open(path).convert('L')
    if np.array(img).mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28))
    x = np.array(img).astype('float32') / 255.0
    x = (x - 0.1307) / 0.3081
    x = x[None, None, :, :]
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', type=str, default='model.onnx')
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--classes', type=str, default='class_names.json')
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        raise FileNotFoundError(f'{args.onnx} not found.')
    if not os.path.exists(args.classes):
        with open(args.classes, 'w') as f:
            json.dump([str(i) for i in range(10)], f)

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    x = load_image_28x28(args.image)
    inputs = {sess.get_inputs()[0].name: x}
    logits = sess.run(None, inputs)[0]
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum(axis=1, keepdims=True)

    with open(args.classes, 'r') as f:
        class_names = json.load(f)

    top = int(probs.argmax())
    print('Prediction:', class_names[top])
    print('Probabilities:')
    for i, p in enumerate(probs[0]):
        print(f'  {class_names[i]}: {p:.4f}')

if __name__ == '__main__':
    main()
