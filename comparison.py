"""Sinh comparison visualizations vào output/comparison/."""

import os
import warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from utils import load_pkl, makedirs
from dataset import load_dataset, stratified_split
from features import extract_all_features
from encode import build_nn_index, encode_all

os.chdir(Path(__file__).resolve().parent)

OUT = "output/comparison"
makedirs(OUT)

CLASS_NAMES = [
    "airplanes", "Motorbikes", "Faces_easy", "Faces",
    "watch", "Leopards", "bonsai", "car_side", "ketch", "chandelier",
]

# ── Thu thập kết quả từ tất cả checkpoint ────────────────────────────────────
RUNS = [
    ("ORB\nk=100\nknn",        "20260502_184354__orb_k100_knn"),
    ("SIFT\nk=100\nknn",       "20260502_184919__sift_k100_knn"),
    ("SIFT\nk=500\nknn",       "20260502_194617__sift_k500_knn"),
    ("SIFT\nk=1000\nknn",      "20260502_194846__sift_k1000_knn"),
    ("SIFT\nk=1000\nrf",       "20260502_195624__sift_k1000_rf"),
    ("SIFT\nk=1000\nsvm_lin",  "20260502_200302__sift_k1000_svm_linear"),
    ("SIFT\nk=1000\nsvm_rbf",  "20260502_200652__sift_k1000_svm_rbf"),
    ("SIFT\nk=2000\nsvm_rbf",  "20260502_201756__sift_k2000_svm_rbf"),
]

print("Thu thập predictions từ checkpoint...")
results = []  # (label, acc, f1_per_class, y_true, y_pred)

for label, folder in RUNS:
    ckpt = Path(f"output/final/{folder}/model.pkl")
    data = load_pkl(str(ckpt))
    cfg, clf, codebook = data["cfg"], data["clf"], data["codebook"]

    paths, labels, _ = load_dataset(cfg)
    _, _, test_paths, test_labels = stratified_split(paths, labels, cfg)
    test_feats = extract_all_features(test_paths, cfg)
    nn = build_nn_index(codebook)
    X_test = encode_all(test_feats, codebook, nn, cfg)
    y_pred = clf.predict(X_test).tolist()

    acc = (np.array(test_labels) == np.array(y_pred)).mean() * 100
    _, _, f1, _ = precision_recall_fscore_support(
        test_labels, y_pred, labels=list(range(len(CLASS_NAMES)))
    )
    results.append((label, acc, f1, test_labels, y_pred))
    print(f"  {label.replace(chr(10), ' '):<25} {acc:.2f}%")

labels_list = [r[0] for r in results]
accs        = [r[1] for r in results]
f1_matrix   = np.array([r[2] for r in results])  # (n_runs, n_classes)

# ── 1. Accuracy progression bar chart ────────────────────────────────────────
colors = ["#d94f3d" if i == 0 else "#2196F3" if i < 6 else "#4CAF50" if i == 6 else "#FF9800"
          for i in range(len(accs))]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(range(len(accs)), accs, color=colors, width=0.6, edgecolor="white")
ax.set_xticks(range(len(accs)))
ax.set_xticklabels(labels_list, fontsize=8)
ax.set_ylim(40, 90)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy theo từng bước cải tiến", fontsize=13)
ax.axhline(accs[0], color="#d94f3d", linestyle="--", linewidth=0.8, alpha=0.5)

for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{acc:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

legend_patches = [
    mpatches.Patch(color="#d94f3d", label="Baseline (ORB)"),
    mpatches.Patch(color="#2196F3", label="Cải tiến từng bước"),
    mpatches.Patch(color="#4CAF50", label="Best (SIFT k=1000 svm_rbf)"),
    mpatches.Patch(color="#FF9800", label="k=2000 (giảm)"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/1_accuracy_progression.png", dpi=120, bbox_inches="tight")
plt.close()
print("\nSaved: 1_accuracy_progression.png")

# ── 2. F1 heatmap (class × run) ───────────────────────────────────────────────
short_labels = [l.replace("\n", " ") for l in labels_list]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(f1_matrix.T, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(short_labels)))
ax.set_xticklabels(short_labels, fontsize=8)
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_yticklabels(CLASS_NAMES, fontsize=9)
ax.set_title("F1-score mỗi class × mỗi run", fontsize=12)

for i in range(len(short_labels)):
    for j in range(len(CLASS_NAMES)):
        val = f1_matrix[i, j]
        ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="black" if val > 0.3 else "white")

fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
plt.tight_layout()
plt.savefig(f"{OUT}/2_f1_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 2_f1_heatmap.png")

# ── 3. Delta accuracy (incremental gain) ─────────────────────────────────────
deltas = [0] + [accs[i] - accs[i-1] for i in range(1, len(accs))]
delta_colors = ["#4CAF50" if d >= 0 else "#f44336" for d in deltas]
step_labels = [
    "Baseline", "ORB→SIFT", "k100→500", "k500→1000",
    "knn→rf", "rf→svm_lin", "lin→rbf", "k1000→2000",
]

fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(range(len(deltas)), deltas, color=delta_colors, width=0.6, edgecolor="white")
ax.set_xticks(range(len(deltas)))
ax.set_xticklabels(step_labels, fontsize=9)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_ylabel("Δ Accuracy (%)")
ax.set_title("Mức cải thiện tại mỗi bước", fontsize=12)

for bar, d in zip(bars, deltas):
    if d != 0:
        y = bar.get_height() if d > 0 else bar.get_height() - 0.3
        ax.text(bar.get_x() + bar.get_width() / 2, y + (0.1 if d > 0 else -0.5),
                f"{d:+.1f}%", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{OUT}/3_incremental_gain.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 3_incremental_gain.png")

# ── 4. Class difficulty ranking (mean F1 qua tất cả run) ────────────────────
mean_f1_per_class = f1_matrix.mean(axis=0)  # (n_classes,)
order = np.argsort(mean_f1_per_class)
sorted_classes = [CLASS_NAMES[i] for i in order]
sorted_f1      = mean_f1_per_class[order]
bar_colors     = ["#f44336" if f < 0.4 else "#FF9800" if f < 0.6 else "#4CAF50"
                  for f in sorted_f1]

fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.barh(range(len(sorted_classes)), sorted_f1, color=bar_colors, edgecolor="white")
ax.set_yticks(range(len(sorted_classes)))
ax.set_yticklabels(sorted_classes, fontsize=10)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Mean F1 (trung bình qua tất cả run)")
ax.set_title("Độ khó của từng class (thấp = khó hơn)", fontsize=12)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

for bar, f in zip(bars, sorted_f1):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{f:.2f}", va="center", fontsize=9)

legend_patches = [
    mpatches.Patch(color="#f44336", label="Khó (F1 < 0.4)"),
    mpatches.Patch(color="#FF9800", label="Trung bình (0.4–0.6)"),
    mpatches.Patch(color="#4CAF50", label="Dễ (F1 > 0.6)"),
]
ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/4_class_difficulty.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 4_class_difficulty.png")

# ── 5. Confusion diff: best (svm_rbf) - baseline (orb knn) ──────────────────
from sklearn.metrics import confusion_matrix

y_true_base, y_pred_base = results[0][3], results[0][4]   # ORB k=100 knn
y_true_best, y_pred_best = results[6][3], results[6][4]   # SIFT k=1000 svm_rbf

cm_base = confusion_matrix(y_true_base, y_pred_base).astype(float)
cm_best = confusion_matrix(y_true_best, y_pred_best).astype(float)

# Normalize theo row (recall perspective)
cm_base_norm = cm_base / cm_base.sum(axis=1, keepdims=True)
cm_best_norm = cm_best / cm_best.sum(axis=1, keepdims=True)
diff = cm_best_norm - cm_base_norm  # positive = cải thiện (diagonal), negative = worse

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(diff, cmap="RdYlGn", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(len(CLASS_NAMES)))
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(CLASS_NAMES, fontsize=8)
ax.set_title("Confusion diff: SIFT k=1000 svm_rbf − ORB k=100 knn\n(xanh = cải thiện, đỏ = tệ hơn)", fontsize=11)
ax.set_xlabel("Dự đoán")
ax.set_ylabel("Thực")

for i in range(len(CLASS_NAMES)):
    for j in range(len(CLASS_NAMES)):
        val = diff[i, j]
        if abs(val) > 0.05:
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=7, color="black")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{OUT}/5_confusion_diff.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: 5_confusion_diff.png")

print(f"\nTất cả comparison visuals lưu tại: {OUT}/")
