import argparse, os, torch
from modules import build_model
from dataset import build_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="results/min1/best.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = build_loaders(batch_size=args.batch_size)
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    model = build_model().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)["logits"]
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

    print(f"Validation accuracy (using saved checkpoint): {correct/total:.3f}")

if __name__ == "__main__":
    main()
