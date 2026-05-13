"""Phân loại (SVM/RF/KNN)."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from config import Config


def build_classifier(cfg: Config):
    """Tạo classifier theo config."""
    if cfg.classifier == "svm_rbf":
        return SVC(
            kernel="rbf",
            C=cfg.svm_c,
            gamma=cfg.svm_gamma,
            probability=True,
            random_state=cfg.seed,
        )
    if cfg.classifier == "rf":
        return RandomForestClassifier(
            n_estimators=cfg.rf_trees,
            random_state=cfg.seed,
            n_jobs=-1,
        )
    if cfg.classifier == "svm_linear":
        return SVC(
            kernel="linear",
            C=cfg.svm_c,
            probability=True,
            random_state=cfg.seed,
        )
    if cfg.classifier == "knn":
        return KNeighborsClassifier(n_neighbors=cfg.knn_k, n_jobs=-1)

    raise ValueError(f"Classifier không hỗ trợ: '{cfg.classifier}'")
