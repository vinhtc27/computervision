"""
Tải và phân chia dataset Caltech-101.

Caltech-101 là benchmark chuẩn cho nhận dạng đối tượng, gồm 101 class + background.
Mỗi class có từ 31-800 ảnh, độ phân giải trung bình ~300×200 px.
Dataset được dùng rộng rãi trong các paper BoW gốc [Csurka 2004, Lazebnik 2006].
"""

import random
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from config import Config
from utils import set_seed

# Zip chứa caltech-101/101_ObjectCategories.tar.gz — cần extract 2 lần
_DATASET_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"


def _download_caltech101(data_path: Path) -> None:
    """
    Tự động tải Caltech-101 từ Caltech website và giải nén vào data_path.
    Cấu trúc zip: caltech-101/101_ObjectCategories.tar.gz → cần extract 2 lần.
    """
    tmp = data_path.parent / "_caltech_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / "caltech-101.zip"

    print("Dataset chưa có — đang tải Caltech-101 (~130MB)...")

    def _progress(count, block, total):
        pct = min(count * block / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(_DATASET_URL, zip_path, reporthook=_progress)
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Tải thất bại: {e}\nKiểm tra kết nối mạng.")
    print()

    print("  Giải nén zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    tar_path = tmp / "caltech-101" / "101_ObjectCategories.tar.gz"
    if not tar_path.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Không tìm thấy {tar_path} sau khi giải nén zip.")

    print("  Giải nén tar.gz (lần 2)...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(tmp)

    src = tmp / "101_ObjectCategories"
    if not src.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"Không tìm thấy {src} sau khi giải nén tar.gz.")

    shutil.move(str(src), str(data_path))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"  Dataset sẵn sàng tại: {data_path}")


def load_dataset(cfg: Config):
    """
    Tải danh sách đường dẫn ảnh từ data/.
    Nếu chưa có, tự động tải từ Caltech website (~130MB).
    Returns: (image_paths, labels, class_names)
    """
    data_path = Path(cfg.data_dir).resolve()

    if not data_path.exists():
        _download_caltech101(data_path)

    available = {d.name for d in data_path.iterdir() if d.is_dir()}
    missing = [c for c in cfg.classes if c not in available]
    if missing:
        raise ValueError(
            f"Không tìm thấy class: {missing}\n"
            f"Kiểm tra data_dir='{cfg.data_dir}'\n"
            f"Có sẵn: {sorted(available)[:20]}..."
        )

    image_paths, labels = [], []
    for label_idx, class_name in enumerate(cfg.classes):
        imgs = sorted((data_path / class_name).glob("*.jpg"))
        if cfg.images_per_class is not None:
            imgs = imgs[: cfg.images_per_class]
        image_paths.extend(str(p) for p in imgs)
        labels.extend([label_idx] * len(imgs))

    if cfg.images_per_class is None:
        per_class_msg = "tất cả ảnh"
    else:
        per_class_msg = f"≤{cfg.images_per_class} ảnh"
    print(
        f"Dataset: {len(cfg.classes)} classes × {per_class_msg}"
        f" = {len(image_paths)} ảnh"
    )
    return image_paths, labels, list(cfg.classes)


def stratified_split(paths: list, labels: list, cfg: Config):
    """
    Phân chia train/test theo tỷ lệ cfg.test_split, stratified theo class.
    Đảm bảo mỗi class có tỷ lệ test/train đồng đều.
    Returns: (train_paths, train_labels, test_paths, test_labels)
    """
    set_seed(cfg.seed)
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []

    for lbl in sorted(set(labels)):
        idxs = [i for i, l in enumerate(labels) if l == lbl]
        random.shuffle(idxs)
        n_test = max(1, int(len(idxs) * cfg.test_split))
        for i in idxs[:n_test]:
            test_paths.append(paths[i])
            test_labels.append(labels[i])
        for i in idxs[n_test:]:
            train_paths.append(paths[i])
            train_labels.append(labels[i])

    print(f"Train: {len(train_paths)} | Test: {len(test_paths)}")
    return train_paths, train_labels, test_paths, test_labels
