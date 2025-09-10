import argparse, json, os, time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 28x28 -> 14x14
            nn.Conv2d(16, 32, 3, padding=1),# 14x14 -> 14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(32*7*7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./data", help="dataset root")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out", type=str, default=".")
    ap.add_argument("--save-plots", action="store_true", help="save loss.png and acc.png from history")
    args = ap.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(args.data_dir, train=True, transform=tfm, download=True)
    test_ds  = datasets.MNIST(args.data_dir, train=False, transform=tfm, download=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiz = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimiz.zero_grad()
            loss.backward()
            optimiz.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / total
        acc = correct / total
        dt = time.time() - t0
        print(f"[Epoch {epoch}] train_loss={avg_loss:.4f} acc={acc:.4f} time={dt:.1f}s")
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(acc)

        model.eval()
        t_correct, t_total = 0, 0
        t_loss_sum = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                t_loss_sum += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                t_correct += (pred == y).sum().item()
                t_total += y.size(0)
        test_loss = t_loss_sum/t_total
        test_acc = t_correct/t_total
        print(f"         test_loss={test_loss:.4f} acc={test_acc:.4f}")
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    # save artifacts
    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, 'model.pth')
    torch.save(model.state_dict(), model_path)
    with open(os.path.join(args.out, 'class_names.json'), 'w') as f:
        json.dump([str(i) for i in range(10)], f, indent=2)
    with open(os.path.join(args.out, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved model to {model_path}")
    print(f"Saved history to {os.path.join(args.out, 'history.json')}")

    if args.save_plots:
        import matplotlib.pyplot as plt
        # loss
        plt.figure()
        plt.plot(history['train_loss'], label='train_loss')
        plt.plot(history['test_loss'], label='test_loss')
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(os.path.join(args.out, 'loss.png'), bbox_inches='tight')
        plt.close()
        # acc
        plt.figure()
        plt.plot(history['train_acc'], label='train_acc')
        plt.plot(history['test_acc'], label='test_acc')
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(os.path.join(args.out, 'acc.png'), bbox_inches='tight')
        plt.close()
        print("Saved plots: loss.png, acc.png")

if __name__ == '__main__':
    main()
