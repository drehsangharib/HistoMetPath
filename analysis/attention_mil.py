"""
Phase 2.4-B.2: Attention-based MIL (Ilse et al., 2018)

- Trains an attention MIL network on pseudo-slides
- Evaluates on held-out slides
- FIXED: proper handling of numpy.object_ bags
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split


# -------------------------------------------------------------------------
# Ensure project root
# -------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "embeddings")
BATCH_SIZE = 1            # MIL uses 1 bag per batch
EPOCHS = 20
LR = 1e-4
THRESHOLD = 0.15
TEST_SIZE = 0.30
RANDOM_STATE = 42
DEVICE = "cpu"


# -------------------------------------------------------------------------
# Load pseudo-slides
# -------------------------------------------------------------------------
slide_embeddings = np.load(
    os.path.join(EMBEDDING_DIR, "train_slide_embeddings.npy"),
    allow_pickle=True,
)
slide_labels = np.load(
    os.path.join(EMBEDDING_DIR, "train_slide_labels.npy"),
)

print("Slides:", slide_embeddings.shape[0])


# -------------------------------------------------------------------------
# Dataset (FIXED)
# -------------------------------------------------------------------------
class SlideDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags = bags
        self.labels = labels

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        # ✅ Convert object array → float array → tensor
        bag_np = np.asarray(self.bags[idx], dtype=np.float32)
        bag = torch.from_numpy(bag_np)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return bag, label


# -------------------------------------------------------------------------
# Train / validation split
# -------------------------------------------------------------------------
bags_tr, bags_val, y_tr, y_val = train_test_split(
    slide_embeddings,
    slide_labels,
    test_size=TEST_SIZE,
    stratify=slide_labels,
    random_state=RANDOM_STATE,
)

train_ds = SlideDataset(bags_tr, y_tr)
val_ds   = SlideDataset(bags_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------------------------------------------------------
# Attention MIL model
# -------------------------------------------------------------------------
class AttentionMIL(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, bag):
        # bag: [N, 512]
        A = self.attention(bag)          # [N, 1]
        A = torch.softmax(A, dim=0)      # attention weights
        z = torch.sum(A * bag, dim=0)    # weighted sum
        logit = self.classifier(z)
        return logit, A


model = AttentionMIL().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for bag, label in train_loader:
        bag = bag.squeeze(0).to(DEVICE)
        label = label.to(DEVICE)

        logit, _ = model(bag)
        loss = criterion(logit.view(-1), label.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
model.eval()
all_probs = []
all_targets = []

with torch.no_grad():
    for bag, label in val_loader:
        bag = bag.squeeze(0).to(DEVICE)
        logit, _ = model(bag)
        prob = torch.sigmoid(logit).item()

        all_probs.append(prob)
        all_targets.append(label.item())

all_probs = np.array(all_probs)
all_targets = np.array(all_targets)

auc = roc_auc_score(all_targets, all_probs)
preds = (all_probs >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(all_targets, preds).ravel()
sensitivity = tp / (tp + fn + 1e-8)
specificity = tn / (tn + fp + 1e-8)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\nAttention MIL (validation):")
print(f"  AUC: {auc:.3f}")
print(f"  Sensitivity: {sensitivity:.3f}")
print(f"  Specificity: {specificity:.3f}")
print(f"  Accuracy: {accuracy:.3f}")