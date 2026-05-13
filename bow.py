"""Pipeline BoW cho bộ dữ liệu Caltech-101."""

import argparse
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from config import Config
from utils import set_seed, makedirs, save_pkl, load_pkl, section
from dataset import load_dataset, stratified_split
from preprocess import plot_sample_images
from features import extract_all_features, plot_keypoints, plot_keypoint_counts
from vocab import build_vocabulary, plot_vocabulary_patches
from encode import (
    build_nn_index,
    encode_all,
    plot_bow_histograms,
    plot_bow_sparsity,
    plot_class_similarity,
)
from classify import build_classifier
from eval import evaluate, plot_confusion_matrix, plot_per_class_metrics


def _run_tag(cfg: Config) -> str:
    """Tạo tag từ các tham số chính."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}__{cfg.feature}_k{cfg.vocab_size}_{cfg.classifier}"


def run_pipeline(cfg: Config) -> None:
    os.chdir(Path(__file__).resolve().parent)
    set_seed(cfg.seed)
    tag = _run_tag(cfg)
    out_final = f"{cfg.out_dir}/final/{tag}"
    makedirs(out_final)
    ckpt_path = f"{out_final}/model.pkl"

    # 1. Dataset
    section("1. Tải dataset")
    paths, labels, class_names = load_dataset(cfg)
    train_paths, train_labels, test_paths, test_labels = stratified_split(paths, labels, cfg)

    plot_sample_images(paths, labels, class_names, out_final)

    # 2. Pipeline
    section("2. Pipeline")
    print(
        f"  feature={cfg.feature} | vocab={cfg.vocab_size} | clf={cfg.classifier}"
    )

    section("2a. Trích đặc trưng")
    train_feats = extract_all_features(train_paths, cfg)
    test_feats  = extract_all_features(test_paths, cfg)
    plot_keypoints(train_paths, train_labels, class_names, cfg, out_final)
    plot_keypoint_counts(train_feats, train_labels, class_names, out_final)

    section("2b. Xây dựng từ điển ảo")
    codebook = build_vocabulary(train_feats, cfg)
    nn = build_nn_index(codebook)
    plot_vocabulary_patches(codebook, train_feats, train_paths, cfg, out_final)

    section("2c. Mã hóa BoW")
    X_train = encode_all(train_feats, codebook, nn, cfg)
    X_test = encode_all(test_feats, codebook, nn, cfg)
    print(f"  Feature dim: {X_train.shape[1]} | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    plot_bow_histograms(X_train, train_labels, class_names, cfg, out_final)
    plot_bow_sparsity(X_train, out_final)
    plot_class_similarity(X_train, train_labels, class_names, out_final)

    section("2d. Huấn luyện classifier")
    clf = build_classifier(cfg)
    t0 = time.time()
    clf.fit(X_train, train_labels)
    print(f"  Train: {time.time() - t0:.1f}s")

    section("2e. Đánh giá")
    y_pred = clf.predict(X_test)
    acc = evaluate(test_labels, y_pred, class_names, out_final)
    plot_confusion_matrix(test_labels, y_pred, class_names, out_final)
    plot_per_class_metrics(test_labels, y_pred, class_names, out_final)

    # Lưu checkpoint
    save_pkl({"clf": clf, "codebook": codebook, "cfg": cfg}, ckpt_path)
    print(f"\nTag:        {tag}")
    print(f"Checkpoint: {Path(ckpt_path).name}")
    print(f"Kết quả:    output/final/{tag}/")
    print(f"\nFinal accuracy: {acc * 100:.2f}%")


def parse_args():
    p = argparse.ArgumentParser(description="BoW Object Recognition - Caltech-101")

    p.add_argument("--feature",  choices=["sift", "orb"], help="Chọn feature extractor")
    p.add_argument("--vocab-size", type=int, help="Kích thước vocabulary")
    p.add_argument(
        "--classifier",
        choices=["svm_rbf", "svm_linear", "rf", "knn"],
        help="Chọn classifier",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()

    if args.feature:
        cfg.feature = args.feature
    if args.vocab_size:
        cfg.vocab_size = args.vocab_size
    if args.classifier:
        cfg.classifier = args.classifier

    run_pipeline(cfg)
