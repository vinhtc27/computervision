"""Đánh giá và visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from utils import makedirs


def evaluate(y_true: list, y_pred: list, class_names: list, out_dir: str) -> float:
    """Tính accuracy và lưu classification report."""
    acc = (np.array(y_true) == np.array(y_pred)).mean()
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\nAccuracy: {acc * 100:.2f}%\n{report}")

    makedirs(out_dir)
    with open(f"{out_dir}/5_classification_report.txt", "w") as f:
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n{report}")

    return acc


def plot_confusion_matrix(y_true: list, y_pred: list,
                          class_names: list, out_dir: str) -> None:
    """Vẽ confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Ma trận nhầm lẫn", fontsize=13, pad=10)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/5_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 5_confusion_matrix.png")


def plot_per_class_metrics(y_true: list, y_pred: list,
                           class_names: list, out_dir: str) -> None:
    """Grouped bar chart Precision / Recall / F1 mỗi class."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(class_names))))

    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, p, w, label="Precision", color="steelblue")
    ax.bar(x,     r, w, label="Recall",    color="tomato")
    ax.bar(x + w, f, w, label="F1",        color="seagreen")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 mỗi lớp", fontsize=12)
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/5_per_class_metrics.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 5_per_class_metrics.png")
