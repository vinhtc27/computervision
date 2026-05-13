"""
Demo BoW inference: chọn 2 ảnh bất kỳ, giải thích pipeline và phán đoán.

Dùng model đã train sẵn (SIFT + k=1000 + SVM RBF, accuracy=81.98%).

Usage:
    python test.py                          # random 2 ảnh từ 2 class khác nhau
    python test.py --img1 airplanes --img2 ketch   # chỉ định class
    python test.py --img1 path/to/a.jpg --img2 path/to/b.jpg  # chỉ định file
"""

import argparse
import pickle
import random
import sys
from pathlib import Path
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "output/final/20260502_200652__sift_k1000_svm_rbf/model.pkl"
DATA_DIR   = Path("data")
OUT_PATH   = "output/test_demo.png"

CLASSES = [
    "airplanes", "Motorbikes", "Faces_easy", "Faces",
    "watch", "Leopards", "bonsai", "car_side", "ketch", "chandelier",
]


# ── Pipeline helpers ─────────────────────────────────────────────────────────

def load_model(path: str):
    with open(path, "rb") as f:
        m = pickle.load(f)
    return m["clf"], m["codebook"], m["cfg"]


def preprocess(img_path: str, img_size=(128, 128)):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {img_path}")
    img_bgr = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_AREA)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb, gray


def extract_sift(gray: np.ndarray, cfg):
    det = cv2.SIFT_create(
        nfeatures=cfg.sift_nfeatures,
        contrastThreshold=cfg.sift_contrast_threshold,
        edgeThreshold=cfg.sift_edge_threshold,
    )
    kps, descs = det.detectAndCompute(gray, None)
    return kps, descs


def encode_bow(descs: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    k = len(codebook)
    hist = np.zeros(k, dtype=np.float32)
    if descs is None or len(descs) == 0:
        return hist
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(codebook)
    _, idx = nn.kneighbors(descs)
    # Gán mỗi descriptor vào codeword gần nhất (hard assignment)
    np.add.at(hist, idx.ravel(), 1)
    return hist


def predict(clf, hist_norm: np.ndarray):
    proba  = clf.predict_proba(hist_norm.reshape(1, -1))[0]
    pred   = int(clf.predict(hist_norm.reshape(1, -1))[0])
    scores = clf.decision_function(hist_norm.reshape(1, -1))[0]
    return pred, proba, scores


# ── Image selection ──────────────────────────────────────────────────────────

def resolve_image(arg: str) -> tuple[str, str]:
    """Trả về (img_path, true_class). arg là tên class hoặc đường dẫn file."""
    p = Path(arg)
    if p.exists() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
        # arg là file path — suy class từ tên folder cha nếu có thể
        parent = p.parent.name
        true_cls = parent if parent in CLASSES else "unknown"
        return str(p), true_cls

    # arg là tên class
    if arg not in CLASSES:
        raise ValueError(f"Class '{arg}' không có trong {CLASSES}")
    imgs = list((DATA_DIR / arg).glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong data/{arg}/")
    return str(random.choice(imgs)), arg


def pick_two_random() -> tuple[tuple[str, str], tuple[str, str]]:
    cls1, cls2 = random.sample(CLASSES, 2)
    return resolve_image(cls1), resolve_image(cls2)


# ── Visualization ────────────────────────────────────────────────────────────

def draw_keypoints_rgb(img_rgb: np.ndarray, kps) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    out  = cv2.drawKeypoints(
        gray, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def make_figure(
    img1_rgb, kp_img1, hist1, proba1, pred1, true1,
    img2_rgb, kp_img2, hist2, proba2, pred2, true2,
    kps1, kps2,
):
    correct1 = CLASSES[pred1] == true1
    correct2 = CLASSES[pred2] == true2

    fig = plt.figure(figsize=(18, 20))
    fig.patch.set_facecolor("#0e1117")
    gs  = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.35)

    label_color = lambda ok: "#2ecc71" if ok else "#e74c3c"

    # ── Tiêu đề lớn ──────────────────────────────────────────────────────────
    fig.suptitle(
        "BoW Pipeline Demo — SIFT + k=1000 + SVM RBF",
        fontsize=15, color="white", fontweight="bold", y=0.98,
    )

    # ── 1. Ảnh gốc ───────────────────────────────────────────────────────────
    for col, (img, true, pred, ok) in enumerate([
        (img1_rgb, true1, pred1, correct1),
        (img2_rgb, true2, pred2, correct2),
    ]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img)
        ax.set_title(
            f"Ảnh gốc  |  Class thật: [{true}]",
            color="white", fontsize=10,
        )
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(label_color(ok))
            spine.set_linewidth(3)

    # ── 2. SIFT Keypoints ─────────────────────────────────────────────────────
    for col, (kp_img, kps, true) in enumerate([
        (kp_img1, kps1, true1),
        (kp_img2, kps2, true2),
    ]):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(kp_img)
        ax.set_title(
            f"SIFT Keypoints — {len(kps)} điểm đặc trưng\n"
            f"Mỗi vòng tròn = 1 keypoint (vị trí + scale + orientation)",
            color="#aaaaaa", fontsize=9,
        )
        ax.axis("off")

    # ── 3. BoW Histogram ─────────────────────────────────────────────────────
    k = len(hist1)
    for col, (hist, true, clr) in enumerate([
        (hist1, true1, "#3498db"),
        (hist2, true2, "#e67e22"),
    ]):
        ax = fig.add_subplot(gs[2, col])
        ax.bar(range(k), hist, width=1.0, color=clr, alpha=0.85)
        nonzero = (hist > 0).sum()
        ax.set_title(
            f"BoW Histogram — [{true}]\n"
            f"{nonzero}/{k} codewords active ({nonzero/k*100:.1f}% non-zero)",
            color="#aaaaaa", fontsize=9,
        )
        ax.set_xlabel("Codeword index (visual word)", color="#888888", fontsize=8)
        ax.set_ylabel("Tần số (đã L2-norm)", color="#888888", fontsize=8)
        ax.tick_params(colors="#666666", labelsize=7)
        ax.set_facecolor("#1a1d27")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # ── 4. Top discriminative codewords ──────────────────────────────────────
    ax = fig.add_subplot(gs[3, :])
    diff  = hist1 - hist2
    top_n = 30
    top_idx   = np.argsort(np.abs(diff))[::-1][:top_n]
    top_idx   = top_idx[np.argsort(top_idx)]   # sắp xếp lại theo index
    diff_vals = diff[top_idx]

    colors = ["#3498db" if v > 0 else "#e67e22" for v in diff_vals]
    bars = ax.bar(range(top_n), diff_vals, color=colors, alpha=0.9)
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([str(i) for i in top_idx], rotation=45, fontsize=7, color="#aaaaaa")
    ax.set_title(
        f"Top {top_n} codewords phân biệt nhất (diff = hist[{true1}] − hist[{true2}])\n"
        f"  Xanh = codeword mạnh hơn ở [{true1}]  |  Cam = codeword mạnh hơn ở [{true2}]",
        color="#aaaaaa", fontsize=9,
    )
    ax.set_ylabel("Δ tần số (L2-norm)", color="#888888", fontsize=8)
    ax.set_facecolor("#1a1d27")
    ax.tick_params(axis="y", colors="#666666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    # ── 5. Xác suất dự đoán ──────────────────────────────────────────────────
    for col, (proba, pred, true, ok) in enumerate([
        (proba1, pred1, true1, correct1),
        (proba2, pred2, true2, correct2),
    ]):
        ax = fig.add_subplot(gs[4, col])
        bar_colors = []
        for i, cls in enumerate(CLASSES):
            if i == pred and cls == true:
                bar_colors.append("#2ecc71")   # đúng
            elif i == pred:
                bar_colors.append("#e74c3c")   # sai
            elif cls == true:
                bar_colors.append("#f39c12")   # đúng nhưng không predict
            else:
                bar_colors.append("#555577")

        ax.barh(range(len(CLASSES)), proba * 100, color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(CLASSES)))
        ax.set_yticklabels(CLASSES, fontsize=8, color="#cccccc")
        ax.set_xlabel("Xác suất (%)", color="#888888", fontsize=8)

        result_text = "✓ ĐÚNG" if ok else "✗ SAI"
        ax.set_title(
            f"Phán đoán: [{CLASSES[pred]}]  {result_text}\n"
            f"Xác suất cao nhất: {proba[pred]*100:.1f}%  |  Class thật: [{true}]",
            color=label_color(ok), fontsize=9, fontweight="bold",
        )
        ax.set_facecolor("#1a1d27")
        ax.tick_params(colors="#666666", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

        # Nhãn % trên bar
        for i, p in enumerate(proba):
            if p > 0.03:
                ax.text(p * 100 + 0.5, i, f"{p*100:.1f}%",
                        va="center", fontsize=7, color="white")

    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    print(f"\nSaved: {OUT_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BoW demo — 2 ảnh bất kỳ")
    parser.add_argument("--img1", default=None, help="Class hoặc path ảnh 1")
    parser.add_argument("--img2", default=None, help="Class hoặc path ảnh 2")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Load model: {MODEL_PATH}")
    clf, codebook, cfg = load_model(MODEL_PATH)
    print(f"  Feature={cfg.feature}  vocab={cfg.vocab_size}  clf={cfg.classifier}")

    # ── Chọn ảnh ─────────────────────────────────────────────────────────────
    if args.img1 is None and args.img2 is None:
        (path1, true1), (path2, true2) = pick_two_random()
    elif args.img1 and args.img2:
        path1, true1 = resolve_image(args.img1)
        path2, true2 = resolve_image(args.img2)
    else:
        parser.error("Cung cấp cả --img1 và --img2, hoặc không cung cấp gì cả.")

    print(f"\nẢnh 1: {Path(path1).name}  (class: {true1})")
    print(f"Ảnh 2: {Path(path2).name}  (class: {true2})")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    results = []
    for path, true in [(path1, true1), (path2, true2)]:
        print(f"\n── {true} ──")

        img_rgb, gray = preprocess(path, cfg.img_size)
        print(f"  Preprocess: {cfg.img_size[0]}×{cfg.img_size[1]} grayscale")

        kps, descs = extract_sift(gray, cfg)
        n_kp = len(kps) if kps else 0
        print(f"  SIFT: {n_kp} keypoints, {0 if descs is None else descs.shape} descriptors")

        hist = encode_bow(descs, codebook)
        print(f"  BoW encode: {(hist > 0).sum()}/{cfg.vocab_size} codewords active")

        hist_norm = normalize(hist.reshape(1, -1), norm="l2")[0]

        pred, proba, scores = predict(clf, hist_norm)
        print(f"  Predict: {CLASSES[pred]}  (xác suất: {proba[pred]*100:.1f}%)")
        print(f"  Thật:    {true}  → {'✓ ĐÚNG' if CLASSES[pred] == true else '✗ SAI'}")

        kp_img = draw_keypoints_rgb(img_rgb, kps)
        results.append((img_rgb, kp_img, hist_norm, proba, pred, true, kps))

    # ── Visualize ─────────────────────────────────────────────────────────────
    (img1, kp1, h1, p1, pred1, t1, kps1) = results[0]
    (img2, kp2, h2, p2, pred2, t2, kps2) = results[1]

    print("\nVẽ visualization...")
    make_figure(img1, kp1, h1, p1, pred1, t1, img2, kp2, h2, p2, pred2, t2, kps1, kps2)


if __name__ == "__main__":
    main()
