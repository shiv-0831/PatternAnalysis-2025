import argparse, torch, numpy as np
from collections import defaultdict
from modules import build_model
from dataset import build_loaders

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default="results/run1/best.pt")
    ap.add_argument("--dataset", type=str, default="random", choices=["random","adni"])
    ap.add_argument("--batch_size", type=int, default=32)
    # ADNI flags (only used if dataset=adni)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--labels_csv", type=str, default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--plane", type=str, default="axial")
    ap.add_argument("--slice_mode", type=str, default="center_k")
    ap.add_argument("--center_k", type=int, default=32)
    ap.add_argument("--step_s", type=int, default=2)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te = build_loaders(
        dataset=args.dataset,
        data_root=args.data_root, labels_csv=args.labels_csv,
        plane=args.plane, slice_mode=args.slice_mode, center_k=args.center_k, step_s=args.step_s,
        resize_hw=(224,224), val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        seed=args.seed, batch_size=args.batch_size, num_workers=args.num_workers,
        augment=False
    )
    loader = te if (args.dataset == "adni" and te is not None) else va

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    correct = total = 0
    subj_logits = defaultdict(list)
    subj_labels = {}

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, y, sid = batch
            else:
                sid = None
                x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
            x, y = x.to(device), y.to(device)
            logits = model(x)["logits"]
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)

            if sid is not None:
                for i, s in enumerate(sid):
                    subj_logits[s].append(logits[i].cpu().numpy())
                    subj_labels[s] = int(y[i].cpu().item())

    slice_acc = correct / max(total, 1)
    print(f"Slice-level accuracy: {slice_acc:.3f}")

    if subj_logits:
        p_correct = 0
        for s, logit_list in subj_logits.items():
            mlog = np.mean(np.stack(logit_list, axis=0), axis=0)
            pred = int(np.argmax(mlog))
            p_correct += int(pred == subj_labels[s])
        patient_acc = p_correct / len(subj_logits)
        print(f"Patient-level accuracy: {patient_acc:.3f} (subjects={len(subj_logits)})")

if __name__ == "__main__":
    main()
