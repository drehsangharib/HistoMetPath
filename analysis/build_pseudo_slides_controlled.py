"""
Controlled, split-aware synthetic pseudo-slide construction.

Design:
- No cross-split mixing
- No patch reuse within a split/seed
- Balanced positive / negative bags
- Positive bags contain a fixed fraction of positive patches
- Negative bags contain only negative patches

These are synthetic pseudo-slides, NOT native WSIs.
"""

from pathlib import Path
import argparse
import json
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--patches-per-bag", type=int, default=50)
    ap.add_argument("--positive-fraction", type=float, default=0.20)
    ap.add_argument("--bags", type=int, default=None)
    return ap.parse_args()


DEFAULT_BAGS = {
    "train": 200,
    "valid": 50,
    "test": 50,
}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    root = Path(__file__).resolve().parents[1]
    emb_dir = root / "embeddings"

    embeddings = np.load(emb_dir / f"{args.split}_embeddings.npy")
    labels = np.load(emb_dir / f"{args.split}_labels.npy").astype(int)

    if args.bags is None:
        total_bags = DEFAULT_BAGS[args.split]
    else:
        total_bags = args.bags

    if total_bags % 2 != 0:
        raise ValueError("Bag count must be even")

    bags_per_class = total_bags // 2
    p_per_bag = max(1, int(round(args.patches_per_bag * args.positive_fraction)))
    n_per_bag = args.patches_per_bag - p_per_bag

    pos_idx = rng.permutation(np.where(labels == 1)[0])
    neg_idx = rng.permutation(np.where(labels == 0)[0])

    need_pos = bags_per_class * p_per_bag
    need_neg = bags_per_class * args.patches_per_bag + bags_per_class * n_per_bag

    if len(pos_idx) < need_pos or len(neg_idx) < need_neg:
        raise RuntimeError("Not enough patches for requested configuration")

    bags = []
    bag_labels = []
    bag_fractions = []
    used = set()

    pos_cursor = 0
    neg_cursor = 0

    # Negative bags
    for _ in range(bags_per_class):
        sel = neg_idx[neg_cursor:neg_cursor + args.patches_per_bag]
        neg_cursor += args.patches_per_bag
        bags.append(embeddings[sel])
        bag_labels.append(0)
        bag_fractions.append(0.0)
        used.update(sel.tolist())

    # Positive bags
    for _ in range(bags_per_class):
        sel_pos = pos_idx[pos_cursor:pos_cursor + p_per_bag]
        pos_cursor += p_per_bag
        sel_neg = neg_idx[neg_cursor:neg_cursor + n_per_bag]
        neg_cursor += n_per_bag
        sel = np.concatenate([sel_pos, sel_neg])
        sel = rng.permutation(sel)
        bags.append(embeddings[sel])
        bag_labels.append(1)
        bag_fractions.append(labels[sel].mean())
        used.update(sel.tolist())

    if len(used) != total_bags * args.patches_per_bag:
        raise RuntimeError("Patch reuse detected")

    order = rng.permutation(total_bags)
    bags = np.array([bags[i] for i in order], dtype=object)
    bag_labels = np.array([bag_labels[i] for i in order])
    bag_fractions = np.array([bag_fractions[i] for i in order])

    out_dir = (
        emb_dir
        / "pseudo_slides_controlled"
        / args.split
        / f"seed_{args.seed}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "bags.npy", bags)
    np.save(out_dir / "labels.npy", bag_labels)
    np.save(out_dir / "fractions.npy", bag_fractions)

    audit = {
        "split": args.split,
        "seed": args.seed,
        "bags": int(total_bags),
        "negative_bags": int((bag_labels == 0).sum()),
        "positive_bags": int((bag_labels == 1).sum()),
        "patches_per_bag": args.patches_per_bag,
        "positive_fraction_target": args.positive_fraction,
        "fraction_mean": float(bag_fractions.mean()),
        "fraction_min": float(bag_fractions.min()),
        "fraction_max": float(bag_fractions.max()),
    }

    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2)
    )

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
