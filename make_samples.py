import argparse, os, csv
from torchvision import datasets, transforms
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./data")
    ap.add_argument("--out", type=str, default="./samples")
    ap.add_argument("--count", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tfm = transforms.ToTensor()
    test_ds = datasets.MNIST(args.data_dir, train=False, transform=tfm, download=True)

    rows = [("filename", "label")]
    for i in range(args.count):
        img, label = test_ds[i]
        # img is tensor [1,28,28]
        pil = transforms.ToPILImage()(img)
        path = os.path.join(args.out, f"{i}.png")
        pil.save(path)
        rows.append((f"{i}.png", int(label)))
    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"Saved {args.count} images to {args.out}")

if __name__ == "__main__":
    main()
