"""
PatchCamelyon (PCAM) HDF5 Dataset Loader
Phase 2.3-A: Robust Macenko stain normalization (toggleable)

- Safe fallback for low-information patches
- No training-loop changes required
- Publication-grade behavior
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except ImportError:
    cv2 = None


def macenko_normalize(image, alpha=1, beta=0.15, target_stain_matrix=None):
    """
    Robust Macenko stain normalization with safe fallback.

    If normalization fails (low-information patch), the original image
    is returned unchanged.
    """
    if cv2 is None:
        return image

    try:
        img = image.astype(np.float32) + 1.0
        od = -np.log(img / 255.0)

        od_hat = od.reshape((-1, 3))
        od_hat = od_hat[np.all(od_hat > beta, axis=1)]

        # ✅ Fallback: insufficient stain information
        if od_hat.shape[0] < 10:
            return image

        cov = np.cov(od_hat.T)
        if not np.isfinite(cov).all():
            return image

        _, eigvecs = np.linalg.eigh(cov)
        eigvecs = eigvecs[:, ::-1]

        proj = np.dot(od_hat, eigvecs[:, :2])
        phi = np.arctan2(proj[:, 1], proj[:, 0])

        min_phi = np.percentile(phi, alpha)
        max_phi = np.percentile(phi, 100 - alpha)

        v1 = np.dot(eigvecs[:, :2], [np.cos(min_phi), np.sin(min_phi)])
        v2 = np.dot(eigvecs[:, :2], [np.cos(max_phi), np.sin(max_phi)])

        stain_matrix = np.array([v1, v2]).T
        stain_matrix /= np.linalg.norm(stain_matrix, axis=0)

        if target_stain_matrix is None:
            target_stain_matrix = np.array([
                [0.65, 0.70],
                [0.07, 0.99],
                [0.27, 0.11],
            ])

        conc = np.linalg.lstsq(
            stain_matrix,
            od.reshape((-1, 3)).T,
            rcond=None,
        )[0]

        max_conc = np.percentile(conc, 99, axis=1)
        conc /= max_conc[:, None]

        od_norm = np.dot(target_stain_matrix, conc).T
        img_norm = np.exp(-od_norm).reshape(image.shape) * 255.0

        return np.clip(img_norm, 0, 255).astype(np.uint8)

    except Exception:
        # ✅ Absolute safety net
        return image


class PCamHDF5Dataset(Dataset):
    """
    PCAM dataset with optional robust stain normalization.
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform=None,
        use_stain_norm: bool = False,
    ):
        self.x_path = x_path
        self.y_path = y_path
        self.transform = transform
        self.use_stain_norm = use_stain_norm

        with h5py.File(self.x_path, "r") as f:
            self.length = f["x"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.x_path, "r") as fx:
            image = fx["x"][idx]

        with h5py.File(self.y_path, "r") as fy:
            label = fy["y"][idx]

        image = image.astype(np.uint8)

        if self.use_stain_norm:
            image = macenko_normalize(image)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.float32)

        return image, label