"""Trích đặc trưng (SIFT/ORB) + visualization."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import Config
from preprocess import preprocess_image


def kp_array_to_cv2(kp_array: np.ndarray) -> list:
    """Chuyển (N, 4) array [x, y, size, angle] sang list cv2.KeyPoint."""
    return [cv2.KeyPoint(float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            for r in kp_array]


def _kps_to_array(kps) -> np.ndarray:
    """Chuyển list[cv2.KeyPoint] sang (N, 4) array [x, y, size, angle]."""
    if not kps:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(
        [[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in kps],
        dtype=np.float32,
    )


def extract_features(gray: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray | None]:
    """Trích keypoints và descriptors từ ảnh grayscale."""
    if cfg.feature == "sift":
        det = cv2.SIFT_create(
            nfeatures=cfg.sift_nfeatures,
            contrastThreshold=cfg.sift_contrast_threshold,
            edgeThreshold=cfg.sift_edge_threshold,
        )
        kps, descs = det.detectAndCompute(gray, None)
    elif cfg.feature == "orb":
        det = cv2.ORB_create(nfeatures=cfg.orb_nfeatures)
        kps, descs = det.detectAndCompute(gray, None)
        if descs is not None:
            descs = descs.astype(np.float32)
    else:
        raise ValueError(f"Feature không hỗ trợ: '{cfg.feature}'")

    if descs is None or len(descs) == 0:
        return np.empty((0, 4), dtype=np.float32), None

    return _kps_to_array(kps), descs


def extract_all_features(paths: list[str], cfg: Config) -> list[tuple]:
    """Trích features cho danh sách ảnh."""
    results, n_empty = [], 0
    empty_paths = []

    for path in paths:
        gray = preprocess_image(path, cfg)
        if gray is None:
            results.append((np.empty((0, 4), dtype=np.float32), None))
            n_empty += 1
            empty_paths.append(path)
            continue
        kp_array, descs = extract_features(gray, cfg)
        if descs is None:
            n_empty += 1
            empty_paths.append(path)
        results.append((kp_array, descs))

    if n_empty:
        print(f"  Cảnh báo: {n_empty}/{len(paths)} ảnh không có keypoints")
        preview = empty_paths[:10]
        print("  Ảnh không có keypoints (tối đa 10):")
        for p in preview:
            print(f"    - {p}")
        if n_empty > len(preview):
            print(f"    ... và {n_empty - len(preview)} ảnh khác")

    return results


def plot_keypoints(paths: list, labels: list, class_names: list,
                   cfg: Config, out_dir: str) -> None:
    """Vẽ keypoints trên 5 ảnh mẫu từ 5 class đầu."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(
        f"Điểm keypoint - {cfg.feature.upper()} | resize={cfg.img_size[0]}x{cfg.img_size[1]}",
        fontsize=12,
    )

    sample_paths = [paths[labels.index(i)] for i in range(min(5, len(class_names)))]
    for ax, path, cls in zip(axes, sample_paths, class_names[:5]):
        gray = preprocess_image(path, cfg)
        if gray is None:
            ax.set_title(f"{cls}\n(lỗi ảnh)", fontsize=9)
            ax.axis("off")
            continue
        kp_array, _ = extract_features(gray, cfg)
        kps_cv2 = kp_array_to_cv2(kp_array)
        img_kp = cv2.drawKeypoints(
            gray, kps_cv2, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        ax.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{cls}\n{len(kps_cv2)} điểm", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_keypoints_{cfg.feature}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 2_keypoints_{cfg.feature}.png")


def plot_keypoint_counts(features: list, labels: list, class_names: list,
                         out_dir: str) -> None:
    """Bar chart số keypoints trung bình mỗi class."""
    counts_per_class = {cls: [] for cls in class_names}
    for (kp_array, _), lbl in zip(features, labels):
        counts_per_class[class_names[lbl]].append(len(kp_array))

    means = [np.mean(counts_per_class[cls]) for cls in class_names]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(class_names, means, color="steelblue")
    ax.set_title("Số keypoint trung bình mỗi lớp", fontsize=12)
    ax.set_ylabel("Số keypoint")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/2_keypoint_counts.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 2_keypoint_counts.png")
