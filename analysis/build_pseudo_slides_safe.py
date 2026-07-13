"""
Leakage-safe pseudo-slide builder.

- No cross-split mixing
- Deterministic bagging
- Supports contiguous and shuffled strategies
"""

from pathlib import Path
import json
import numpy as np
import argparse


def build_bags(
    patch_embeddings,
    patch_labels,
    patches_per_bag,
    strategy,
    seed,
):
    rng = np.random.default_rng(seed)

    indices = np.arange(len(patch_labels))

    if strategy == "shuffled":
        rng.shuffle(indices)

    usable = (len(indices) // patches_per_bag) * patches_per_bag
    indices = indices[:usable]

    bags = []
    bag_labels = []
    fractions = []

    for i in range(0, usable, patches_per_bag):
        bag_idx = indices[i : i + patches_per_bag]
        bag = patch_embeddings[bag_idx]
        labels = patch_labels[bag_idx]

        bags.append(bag)
        bag_labels.append(int(labels.max()))
        fractions.append(float(labels.mean()))

    return (
        np.array(bags, dtype=object),
        np.array(bag_labels, dtype=int),
        np.array(fractions),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--strategy", required=True, choices=["contiguous", "shuffled"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patches-per-bag", type=int, default=50)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    emb_dir = root / "embeddings"
    out_dir = emb_dir / "pseudo_slides_safe" / args.split / args.strategy / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_embeddings = np.load(
        emb_dir / f"{args.split}_embeddings.npy",
        allow_pickle=True,
    )
    patch_labels = np.load(
        emb_dir / f"{args.split}_labels.npy",
        allow_pickle=False,
    ).astype(int)

    bags, bag_labels, fractions = build_bags(
        patch_embeddings,
        patch_labels,
        args.patches_per_bag,
        args.strategy,
        args.seed,
    )

    np.save(out_dir / "bags.npy", bags)
    np.save(out_dir / "labels.npy", bag_labels)

    audit = {
        "split": args.split,
        "strategy": args.strategy,
        "seed": args.seed,
        "bags": int(len(bag_labels)),
        "patches_per_bag": args.patches_per_bag,
        "bag_positive_fraction_mean": float(fractions.mean()),
        "bag_positive_fraction_min": float(fractions.min()),
        "bag_positive_fraction_max": float(fractions.max()),
        "pure_negative": int((fractions == 0).sum()),
        "pure_positive": int((fractions == 1).sum()),
        "mixed": int(((fractions > 0) & (fractions < 1)).sum()),
    }

    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
