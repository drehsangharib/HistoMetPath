"""
Phase 2.4-A.2: Construct pseudo-slides (synthetic MIL bags) from PCAM patches.

- Groups patch embeddings into fixed-size bags
- Assigns slide-level labels using MIL assumption
"""

import os
import sys
import numpy as np

# -------------------------------------------------------------------------
# ✅ Ensure project root on PYTHONPATH
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embeddings")
PATCHES_PER_SLIDE = 50   # typical values: 25–100

# -------------------------------------------------------------------------
# Load embeddings (ABSOLUTE PATHS)
# -------------------------------------------------------------------------
embeddings_path = os.path.join(EMBEDDING_DIR, "train_embeddings.npy")
labels_path = os.path.join(EMBEDDING_DIR, "train_labels.npy")

if not os.path.exists(embeddings_path):
    raise FileNotFoundError(f"Missing file: {embeddings_path}")
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Missing file: {labels_path}")

embeddings = np.load(embeddings_path)
labels = np.load(labels_path)

assert embeddings.shape[0] == labels.shape[0]

num_patches = embeddings.shape[0]
num_slides = num_patches // PATCHES_PER_SLIDE

print(f"Total patches: {num_patches}")
print(f"Pseudo-slides: {num_slides}")
print(f"Patches per slide: {PATCHES_PER_SLIDE}")

# -------------------------------------------------------------------------
# Build slide bags
# -------------------------------------------------------------------------
slide_embeddings = []
slide_labels = []

for i in range(num_slides):
    start = i * PATCHES_PER_SLIDE
    end = start + PATCHES_PER_SLIDE

    bag_embeds = embeddings[start:end]
    bag_labels = labels[start:end]

    # MIL assumption: slide positive if ANY patch positive
    slide_label = int(bag_labels.max())

    slide_embeddings.append(bag_embeds)
    slide_labels.append(slide_label)

# Use dtype=object because bags are 2D arrays
slide_embeddings = np.array(slide_embeddings, dtype=object)
slide_labels = np.array(slide_labels)

print("Slide embeddings shape:", slide_embeddings.shape)
print("Slide labels shape:", slide_labels.shape)

# -------------------------------------------------------------------------
# Save pseudo-slides
# -------------------------------------------------------------------------
np.save(os.path.join(EMBEDDING_DIR, "train_slide_embeddings.npy"), slide_embeddings)
np.save(os.path.join(EMBEDDING_DIR, "train_slide_labels.npy"), slide_labels)

print("✅ Pseudo-slide bags saved to:", EMBEDDING_DIR)