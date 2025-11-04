# train.py
import argparse, os, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from modules import build_model, count_params
from dataset import build_loaders

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def plot_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Loss
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png")); plt.close()
    # Acc
    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["val_acc"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_curve.png")); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--head_dropout", type=float, default=0.0)
    ap.add_argument("--save_dir", type=str, default="results/min1")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_loader, val_loader = build_loaders(batch_size=args.batch_size, seed=args.seed)

    model = build_model(in_chans=1, num_classes=2, head_dropout=args.head_dropout).to(device)
    print(f"Model params: {count_params(model):,}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = -1.0, None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        tr_loss, tr_acc, n_tr = 0.0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)["logits"]
            loss = criterion(out, y)
            optim.zero_grad(); loss.backward(); optim.step()
            bs = y.size(0)
            tr_loss += loss.item() * bs
            tr_acc  += accuracy_from_logits(out, y) * bs
            n_tr    += bs

        # val
        model.eval()
        va_loss, va_acc, n_va = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)["logits"]
                loss = criterion(out, y)
                bs = y.size(0)
                va_loss += loss.item() * bs
                va_acc  += accuracy_from_logits(out, y) * bs
                n_va    += bs

        tr_loss /= n_tr; tr_acc /= n_tr
        va_loss /= n_va; va_acc /= n_va
        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {"state_dict": model.state_dict(), "epoch": epoch}
            torch.save(best_state, os.path.join(args.save_dir, "best.pt"))

    plot_curves(history, args.save_dir)
    print(f"Best val_acc: {best_val_acc:.3f} (checkpoint saved)")

if __name__ == "__main__":
    main()
