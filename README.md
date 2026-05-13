# Bag-of-Words (BoW) trên Caltech-101

Triển khai pipeline nhận dạng đối tượng BoW cổ điển trên bộ dữ liệu Caltech-101 (10 class, ~3.300 ảnh).

---

## Pipeline

```text
Ảnh đầu vào
    │
    ▼  [1] Tiền xử lý      resize 128×128 + grayscale
    │
    ▼  [2] Trích đặc trưng  SIFT hoặc ORB → descriptors
    │
    ▼  [3] Visual vocabulary  K-Means → codebook k codeword
    │
    ▼  [4] Mã hóa BoW       hard assignment → histogram → L2 normalize
    │
    ▼  [5] Phân loại        KNN / RF / SVM
    │
    ▼  Accuracy + trực quan hóa
```

---

## Kỹ thuật ở từng bước

### Bước 2 — Trích đặc trưng

| | SIFT | ORB |
| --- | --- | --- |
| **Descriptor** | 128-dim float (gradient histogram) | 32-byte binary (BRIEF) |
| **Keypoint** | DoG (Difference of Gaussian) | FAST corner |
| **Bất biến** | scale + rotation + lighting | rotation (limited scale) |
| **Tốc độ** | chậm (~5–10× ORB) | nhanh |
| **Phân biệt** | cao | trung bình |
| **Dùng khi** | cần accuracy | cần tốc độ / baseline |

SIFT xây dựng pyramid Gaussian đa tầng → keypoint ổn định hơn, descriptor mô tả gradient chi tiết hơn → phân biệt tốt hơn trong không gian BoW.
ORB dùng binary descriptor → khoảng cách Hamming thay vì Euclidean, nhưng khi cast sang float32 để dùng KD-tree thì mất lợi thế tốc độ tính khoảng cách.

### Bước 3 — Visual Vocabulary (K-Means)

Gom toàn bộ descriptors từ tập train thành `k` centroid (codeword). Mỗi codeword = một "từ hình ảnh" đặc trưng.

| Vocab size `k` | Codebook | Hiệu quả |
| --- | --- | --- |
| 100 | thô, nhiều collision | underfitting, phân biệt kém |
| 500 | cân bằng | điểm ngọt cho hầu hết bộ dữ liệu |
| 1000 | mịn | phân biệt tốt hơn, train lâu hơn |
| 2000 | quá mịn | overfitting, sparse histogram |

Tăng `k` giúp đến một giới hạn nhất định; sau đó codebook quá mịn dẫn đến variance cao.

### Bước 4 — Mã hóa BoW

**Hard assignment**: mỗi descriptor gán vào 1 codeword gần nhất (nearest neighbor) → tích lũy histogram tần suất → chuẩn hóa L2.

L2 normalize loại bỏ ảnh hưởng của số lượng keypoint (ảnh nhiều texture ≠ ảnh ít texture nhưng cùng nội dung).

### Bước 5 — Phân loại

| Classifier | Nguyên lý | Ưu điểm | Nhược điểm |
| --- | --- | --- | --- |
| **KNN** (`knn`) | vote từ k láng giềng gần nhất trong không gian histogram | không cần train | chậm khi predict, nhạy noise, kém trong không gian thưa chiều cao |
| **Random Forest** (`rf`) | ensemble 200 cây quyết định, vote đa số | không cần scaling, robust | chậm hơn SVM, bộ nhớ lớn |
| **SVM Linear** (`svm_linear`) | tìm hyperplane phân tách tối ưu, kernel tuyến tính | nhanh, tốt cho feature thưa chiều cao | kém hơn RBF với BoW phi tuyến |
| **SVM RBF** (`svm_rbf`) | mapping lên không gian cao chiều bằng RBF kernel | mạnh nhất cho BoW: L2-normalized histogram + RBF ≈ χ² kernel | chậm hơn linear, cần tune C và gamma |

SVM RBF phù hợp nhất với BoW vì histogram L2-normalized trong không gian RBF gần tương đương χ² kernel — đây là metric chuẩn để so sánh histogram.

---

## Cài đặt

```bash
make install
```

Dataset tự động tải từ Caltech website (~130MB) lần đầu chạy, lưu vào `data/`.

---

## Chạy

```bash
make baseline                                          # ORB + k=100 + KNN  (yếu nhất)
make optimal                                           # SIFT + k=1000 + SVM RBF  (mạnh nhất)
make run                                               # default config (= baseline)
make run ARGS="--feature sift --vocab-size 500 --classifier svm_rbf"
make run ARGS="--feature orb  --vocab-size 500 --classifier rf"
```

---

## Cấu trúc thư mục

```text
.
├── bow.py          # Entry point + pipeline orchestrator
├── config.py       # Dataclass Config (tất cả hyperparameter)
├── dataset.py      # Tải + split Caltech-101
├── preprocess.py   # Bước 1: resize + grayscale
├── features.py     # Bước 2: SIFT/ORB + visualization
├── vocab.py        # Bước 3: K-Means codebook + patch visualization
├── encode.py       # Bước 4: hard assignment + BoW histogram
├── classify.py     # Bước 5: SVM / RF / KNN
├── eval.py         # Metrics + confusion matrix
├── utils.py        # I/O helpers
├── data/           # Ảnh Caltech-101 (tự động tải)
└── output/final/   # Kết quả mỗi lần chạy (tag theo timestamp)
```

---

## Output

Mỗi lần chạy tạo thư mục `output/final/<timestamp>__<feature>_k<vocab>_<clf>/`:

| File | Mô tả |
| --- | --- |
| `1_sample_images.png` | Ảnh mẫu mỗi class |
| `2_keypoints_<feature>.png` | Keypoints trên 5 ảnh mẫu |
| `2_keypoint_counts.png` | Số keypoint trung bình mỗi class |
| `3_visual_vocabulary_patches.png` | Patch ảnh gần centroid nhất mỗi visual word |
| `4_bow_histograms.png` | BoW histogram trung bình mỗi class |
| `4_bow_sparsity.png` | Phân bố độ thưa (tỷ lệ non-zero) |
| `4_bow_class_similarity.png` | Cosine similarity giữa mean BoW các class |
| `5_confusion_matrix.png` | Ma trận nhầm lẫn trên test set |
| `5_per_class_metrics.png` | Grouped bar Precision/Recall/F1 mỗi class |
| `5_classification_report.txt` | Accuracy + precision/recall/F1 mỗi class |
| `model.pkl` | Checkpoint: clf + codebook + cfg |

---

## Kết quả benchmark

Sinh comparison visuals:

```bash
.venv/bin/python comparison.py
```

### Comparison visuals — [**→ xem tất cả trong `output/comparison/`**](output/comparison/)

| Visualize | Mô tả |
| --- | --- |
| [**→ Accuracy progression**](output/comparison/1_accuracy_progression.png) | Accuracy từng bước cải tiến |
| [**→ F1 heatmap**](output/comparison/2_f1_heatmap.png) | F1 mỗi class × mỗi run |
| [**→ Incremental gain**](output/comparison/3_incremental_gain.png) | Δ accuracy tại mỗi bước |
| [**→ Class difficulty**](output/comparison/4_class_difficulty.png) | Ranking class theo mean F1 (dễ → khó) |
| [**→ Confusion diff**](output/comparison/5_confusion_diff.png) | Diff confusion matrix: best − baseline |

### Bảng kết quả

| # | Feature | Vocab | Classifier | Accuracy | Δ | Chi tiết |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | ORB | 100 | knn | 53.07% | baseline | [→ report](output/final/20260502_184354__orb_k100_knn/5_classification_report.txt) |
| 2 | SIFT | 100 | knn | 62.57% | +9.50% | [→ report](output/final/20260502_184919__sift_k100_knn/5_classification_report.txt) |
| 3 | SIFT | 500 | knn | 69.70% | +7.13% | [→ report](output/final/20260502_194617__sift_k500_knn/5_classification_report.txt) |
| 4 | SIFT | 1000 | knn | 72.77% | +3.07% | [→ report](output/final/20260502_194846__sift_k1000_knn/5_classification_report.txt) |
| 5 | SIFT | 1000 | rf | 73.56% | +0.79% | [→ report](output/final/20260502_195624__sift_k1000_rf/5_classification_report.txt) |
| 6 | SIFT | 1000 | svm_linear | 80.30% | +6.74% | [→ report](output/final/20260502_200302__sift_k1000_svm_linear/5_classification_report.txt) |
| **7 ★** | **SIFT** | **1000** | **svm_rbf** | **81.98%** | **+1.68%** | [**→ report**](output/final/20260502_200652__sift_k1000_svm_rbf/5_classification_report.txt) |
| 8 | SIFT | 2000 | svm_rbf | 80.30% | −1.68% | [→ report](output/final/20260502_201756__sift_k2000_svm_rbf/5_classification_report.txt) |

### Nhận xét

**Đóng góp của từng bước (incremental):**

- **ORB → SIFT** (+9.5%): cải thiện lớn nhất — feature extractor là yếu tố quan trọng nhất
- **k=100 → 500** (+7.1%): codebook thô gây mất thông tin đáng kể
- **k=500 → 1000** (+3.1%): diminishing returns, nhưng vẫn đáng
- **knn → rf** (+0.8%): ít cải thiện
- **rf → svm_linear** (+6.7%): SVM vượt trội rõ rệt so với tree-based
- **svm_linear → svm_rbf** (+1.7%): RBF nhỉnh hơn linear, phù hợp với BoW histogram
- **k=1000 → 2000** (−1.7%): codebook quá mịn → histogram quá thưa → kém hơn

**Class khó nhất** (F1 thấp nhất nhất quán qua tất cả run): `bonsai`, `ketch`, `chandelier`, `car_side` — 4 class ít ảnh nhất (32–38 test samples).

**Class dễ nhất**: `Motorbikes`, `Leopards`, `Faces_easy` — đặc trưng hình ảnh khác biệt rõ ràng.

**RF vs SVM**: RF precision cao nhưng recall rất thấp ở class nhỏ (bonsai F1=0.10, ketch F1=0.00) — bias bởi class lớn. SVM cân bằng hơn.
