"""Cấu hình cho pipeline BoW."""

from dataclasses import dataclass, field


# 10 class nhiều ảnh nhất trong Caltech-101 (dùng tất cả ảnh mỗi class)
SELECTED_CLASSES = [
    "airplanes",
    "Motorbikes",
    "Faces_easy",
    "Faces",
    "watch",
    "Leopards",
    "bonsai",
    "car_side",
    "ketch",
    "chandelier",
]


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    data_dir: str = "./data"                # thư mục gốc chứa class folders
    out_dir:  str = "./output"              # thư mục lưu tất cả kết quả
    classes:  list = field(default_factory=lambda: SELECTED_CLASSES)
    images_per_class: int | None = None     # None = dùng tất cả ảnh/class
    test_split: float = 0.3                 # 30% test / 70% train (stratified)
    seed: int = 42

    # ── Tiền xử lý ───────────────────────────────────────────────────────────
    img_size:   tuple = (128, 128)          # resize về kích thước cố định (w, h)

    # ── Trích đặc trưng ──────────────────────────────────────────────────────
    feature: str = "orb"                    # "sift" | "orb"
    sift_nfeatures: int = 0                 # 0 = không giới hạn
    sift_contrast_threshold: float = 0.04
    sift_edge_threshold: float = 10.0
    orb_nfeatures: int = 500

    # ── Từ điển ảo ───────────────────────────────────────────────────────────
    vocab_size: int = 100
    kmeans_max_iter: int = 100
    kmeans_n_init: int = 3

    # ── Phân loại ────────────────────────────────────────────────────────────
    classifier: str = "knn"                 # "svm_rbf" | "svm_linear" | "rf" | "knn"
    svm_c: float = 10.0
    svm_gamma: str = "scale"
    rf_trees: int = 200
    knn_k: int = 5

