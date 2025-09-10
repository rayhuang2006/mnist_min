import json, os, argparse
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=str, default="history.json")
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    with open(args.history, "r") as f:
        h = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    plt.figure()
    plt.plot(h["train_loss"], label="train_loss")
    plt.plot(h["test_loss"], label="test_loss")
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "loss.png"), bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(h["train_acc"], label="train_acc")
    plt.plot(h["test_acc"], label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "acc.png"), bbox_inches="tight")
    plt.close()

    print("Saved plots: loss.png, acc.png")

if __name__ == "__main__":
    main()
