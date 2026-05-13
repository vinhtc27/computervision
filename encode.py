"""Mã hóa BoW (hard assignment) + visualization."""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from config import Config


def build_nn_index(codebook: np.ndarray) -> NearestNeighbors:
    """Tạo chỉ mục nearest-neighbor cho codebook."""
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree", n_jobs=-1)
    nn.fit(codebook)
    return nn


def encode_image(kp_array: np.ndarray, descs: np.ndarray, nn: NearestNeighbors,
                 vocab_size: int) -> np.ndarray:
    """Mã hóa 1 ảnh thành histogram BoW (hard assignment)."""
    hist = np.zeros(vocab_size, dtype=np.float32)
    if descs is None or len(descs) == 0:
        return hist
    _, idx = nn.kneighbors(descs)
    np.add.at(hist, idx.ravel(), 1)
    return hist


def encode_all(all_features: list[tuple], codebook: np.ndarray, nn: NearestNeighbors,
               cfg: Config) -> np.ndarray:
    """Mã hóa tất cả ảnh -> matrix (n_images, vocab_size)."""
    k = len(codebook)
    X = [encode_image(kp_array, descs, nn, k) for kp_array, descs in all_features]
    X = np.array(X, dtype=np.float32)
    return normalize(X, norm="l2")


def plot_bow_histograms(X: np.ndarray, labels: list, class_names: list,
                        cfg: Config, out_dir: str) -> None:
    """Vẽ histogram BoW trung bình mỗi class."""
    k = X.shape[1]
    n_cols = 5
    n_rows = (len(class_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9))
    fig.suptitle("BoW Histograms trung bình mỗi lớp", fontsize=12)

    for idx, cls in enumerate(class_names):
        mask = np.array(labels) == idx
        mean_hist = X[mask].mean(axis=0)
        ax = axes[idx // n_cols][idx % n_cols]
        ax.bar(range(k), mean_hist, width=1.0, color=f"C{idx % 10}")
        ax.set_title(cls, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(class_names), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_bow_histograms.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 4_bow_histograms.png")


def plot_bow_sparsity(X: np.ndarray, out_dir: str) -> None:
    """Vẽ phân bố độ thưa (tỷ lệ non-zero) của BoW histograms."""
    nonzero_ratio = (X > 0).sum(axis=1) / X.shape[1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nonzero_ratio, bins=20, color="slateblue", alpha=0.8)
    ax.set_title("Độ thưa của BoW histograms", fontsize=11)
    ax.set_xlabel("Tỷ lệ non-zero")
    ax.set_ylabel("Số ảnh")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_bow_sparsity.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 4_bow_sparsity.png")


def plot_class_similarity(X: np.ndarray, labels: list, class_names: list, out_dir: str) -> None:
    """Heatmap cosine similarity giữa mean BoW của các lớp."""
    means = []
    for idx in range(len(class_names)):
        mask = np.array(labels) == idx
        means.append(X[mask].mean(axis=0))
    means = np.array(means)

    norms = np.linalg.norm(means, axis=1, keepdims=True) + 1e-8
    means = means / norms
    sim = means @ means.T

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Độ tương đồng cosine giữa mean BoW của các lớp", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/4_bow_class_similarity.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 4_bow_class_similarity.png")
