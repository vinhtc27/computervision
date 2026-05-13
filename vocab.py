"""Visual vocabulary (K-Means) + visualization."""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from config import Config


def build_vocabulary(train_features: list[tuple], cfg: Config) -> np.ndarray:
    """Gom descriptors bằng K-Means để tạo codebook."""
    all_descs = np.vstack(
        [d for _, d in train_features if d is not None]
    ).astype(np.float32)

    print(f"  {len(all_descs):,} descriptors -> KMeans(k={cfg.vocab_size})")
    t0 = time.time()

    km = KMeans(
        n_clusters=cfg.vocab_size,
        max_iter=cfg.kmeans_max_iter,
        n_init=cfg.kmeans_n_init,
        random_state=cfg.seed,
    )
    km.fit(all_descs)
    print(f"  Codebook: {km.cluster_centers_.shape} | {time.time() - t0:.1f}s")
    return km.cluster_centers_


def plot_vocabulary_patches(
    codebook: np.ndarray,
    train_features: list[tuple],
    train_paths: list[str],
    cfg,
    out_dir: str,
    n_show: int = 100,
    patch_px: int = 32,
) -> None:
    """Với mỗi visual word, crop patch ảnh gốc gần centroid nhất."""

    # Build lookup: (desc_idx) → (path, kp_x, kp_y, kp_size)
    meta = []
    all_descs_list = []
    for (kp_arr, descs), path in zip(train_features, train_paths):
        if descs is None:
            continue
        for i, kp in enumerate(kp_arr):
            meta.append((path, float(kp[0]), float(kp[1]), float(kp[2])))
            all_descs_list.append(descs[i])

    if not all_descs_list:
        return
    all_descs = np.array(all_descs_list, dtype=np.float32)

    k_show = min(n_show, len(codebook))
    n_cols = 10
    n_rows = (k_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.2, n_rows * 1.2))
    fig.suptitle(
        f"Visual words — patch gần centroid nhất (k={len(codebook)}, hiện {k_show})",
        fontsize=11,
    )

    for word_idx in range(k_show):
        centroid = codebook[word_idx]
        diffs = all_descs - centroid
        nearest = int(np.argmin((diffs * diffs).sum(axis=1)))
        path, cx, cy, size = meta[nearest]

        img = cv2.imread(path)
        if img is None:
            patch = np.zeros((patch_px, patch_px, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, cfg.img_size, interpolation=cv2.INTER_AREA)
            half = patch_px // 2
            x1 = max(0, int(cx) - half)
            y1 = max(0, int(cy) - half)
            x2 = min(img.shape[1], int(cx) + half)
            y2 = min(img.shape[0], int(cy) + half)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                patch = np.zeros((patch_px, patch_px, 3), dtype=np.uint8)
            else:
                patch = cv2.resize(crop, (patch_px, patch_px))
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        ax = axes[word_idx // n_cols][word_idx % n_cols]
        ax.imshow(patch)
        ax.axis("off")

    for idx in range(k_show, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/3_visual_vocabulary_patches.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 3_visual_vocabulary_patches.png")
