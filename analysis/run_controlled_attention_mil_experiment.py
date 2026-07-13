"""
Controlled Attention MIL evaluation on synthetic PCAM pseudo-slides.

Protocol:
- Train on controlled train bags
- Select threshold on validation bags
- Evaluate once on controlled test bags
- Repeat across seeds
- Save metrics and attention statistics

This evaluates synthetic pseudo-slides, not native WSIs.
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)
from analysis.attention_mil_v2 import AttentionMIL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "embeddings" / "pseudo_slides_controlled"
OUT = PROJECT_ROOT / "outputs" / "mil_controlled_attention"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 23, 42, 71, 101]
EPOCHS = 20
LR = 1e-3


def load_split(split, seed):
    d = BASE / split / f"seed_{seed}"
    bags = np.load(d / "bags.npy", allow_pickle=True)
    labels = np.load(d / "labels.npy")
    return bags, labels


def train_attention(train_bags, train_labels, in_dim=512):
    model = AttentionMIL(in_dim=in_dim)
    optimizer = Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(EPOCHS):
        for bag, label in zip(train_bags, train_labels):
            x = torch.from_numpy(np.array(bag.tolist(), dtype=np.float32))
            y = torch.tensor(label, dtype=torch.float32)

            logit, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(
                logit.unsqueeze(0), y.unsqueeze(0)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def predict(model, bags):
    model.eval()
    probs = []
    attentions = []

    with torch.no_grad():
        for bag in bags:
            x = torch.from_numpy(np.array(bag.tolist(), dtype=np.float32))
            logit, A = model(x)
            probs.append(torch.sigmoid(logit).item())
            attentions.append(A.numpy())

    return np.array(probs), attentions


def select_threshold(y, p):
    thresholds = np.unique(p)
    best_t, best = 0.5, -1
    for t in thresholds:
        ba = balanced_accuracy_score(y, p >= t)
        if ba > best:
            best, best_t = ba, t
    return best_t


results = []

for seed in SEEDS:
    print(f"Attention MIL seed={seed}")

    train_bags, train_labels = load_split("train", seed)
    valid_bags, valid_labels = load_split("valid", seed)
    test_bags, test_labels = load_split("test", seed)

    model = train_attention(train_bags, train_labels)

    pv, _ = predict(model, valid_bags)
    thr = select_threshold(valid_labels, pv)

    pt, attentions = predict(model, test_bags)

    results.append({
        "seed": seed,
        "threshold": float(thr),
        "test_auroc": float(roc_auc_score(test_labels, pt)),
        "test_auprc": float(average_precision_score(test_labels, pt)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(test_labels, pt >= thr)
        ),
        "attention_mean_entropy": float(
            np.mean([
                -np.sum(a * np.log(a + 1e-8)) for a in attentions
            ])
        )
    })

OUT.joinpath("attention_results.json").write_text(
    json.dumps(results, indent=2)
)

print(json.dumps(results, indent=2))
print("PASS: Attention MIL controlled evaluation completed.")
