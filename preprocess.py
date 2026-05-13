"""Bước tiền xử lý: resize + grayscale và visualization cơ bản."""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import Config


def preprocess_image(path: str, cfg: Config) -> np.ndarray | None:
    """Đọc ảnh, resize, chuyển grayscale."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, cfg.img_size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def preprocess_batch(paths: list[str], cfg: Config) -> list[np.ndarray | None]:
    """Tiền xử lý một batch ảnh."""
    return [preprocess_image(p, cfg) for p in paths]


def plot_sample_images(paths: list, labels: list, class_names: list, out_dir: str) -> None:
    """Vẽ 1 ảnh mẫu mỗi class (lưới 2x5 cho 10 class)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_cols = 5
    n_rows = (len(class_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    fig.suptitle("Caltech-101 - Ảnh mẫu mỗi lớp", fontsize=14, fontweight="bold")

    for idx, cls in enumerate(class_names):
        cls_paths = [p for p, l in zip(paths, labels) if l == idx]
        if not cls_paths:
            continue
        raw = cv2.imread(cls_paths[0])
        if raw is None:
            continue
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        ax = axes[idx // n_cols][idx % n_cols]
        ax.imshow(img)
        ax.set_title(cls, fontsize=9)
        ax.axis("off")

    for idx in range(len(class_names), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path / "1_sample_images.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 1_sample_images.png")
