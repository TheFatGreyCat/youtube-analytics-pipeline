# YouTube Viral Prediction System

Hệ thống ML end-to-end dự đoán viral potential cho kênh YouTube và video mới đăng.

## Tổng quan

```
Nhập tên kênh → YouTube API + BigQuery → Feature Engineering
                    ↓
              Model A (Channel)  →  Viral probability + Cluster
              Model B (Video)    →  Will viral? + Time window (7d/30d)
                    ↓
              Report: JSON + Print summary
```

---

## Cài đặt

### 1. Cài dependencies

```bash
pip install -r ml/requirements.txt
```

### 2. Cấu hình .env

```bash
cp .env.example .env
# Điền YOUTUBE_API_KEY và GOOGLE_APPLICATION_CREDENTIALS
```

### 3. Đảm bảo có credentials BigQuery

Thả file service account JSON vào thư mục `credentials/`.

---

## Sử dụng

### Training (lần đầu)

```python
from ml.src.pipeline.viral_system import ViralPredictionSystem

system = ViralPredictionSystem()
system.train()         # Load BigQuery → Label → Feature → Cluster → Train A + B
system.save()          # Lưu vào ml/trained_models/
```

### Inference — Dự đoán kênh

```python
from ml.src.pipeline.viral_system import ViralPredictionSystem

system = ViralPredictionSystem.load()
report = system.predict_channel("MrBeast")
report.print_report()
print(report.to_json())
```

### Inference — Dự đoán video mới nhất

```python
report = system.predict_video("MrBeast")        # video mới nhất
report = system.predict_video("MrBeast", video_id="dQw4w9WgXcQ")  # video cụ thể
report.print_report()
```

### Monitor video mới đăng

```python
from ml.src.data.youtube_client import YouTubeAPIClient
from ml.src.pipeline.polling_monitor import PollingMonitor

api = YouTubeAPIClient()
monitor = PollingMonitor(api)

# Bắt đầu poll mỗi 6h trong 72h đầu
monitor.start("VIDEO_ID_HERE", interval_hours=6, duration_hours=72)

# Lấy snapshots
snapshots = monitor.get_snapshots("VIDEO_ID_HERE")
```

---

## Cấu trúc thư mục

```
ml/
├── src/
│   ├── data/
│   │   ├── bigquery_loader.py      ← Load 3 bảng int_ từ BigQuery
│   │   ├── youtube_client.py       ← YouTube API v3 + cache + quota
│   │   └── feature_engineer.py    ← Tất cả feature engineering
│   ├── models/
│   │   ├── label_creator.py        ← Bước 0: tạo labels
│   │   ├── channel_clusterer.py    ← K-Means clustering
│   │   ├── channel_classifier.py   ← Model A (LOOCV)
│   │   ├── video_classifier.py     ← Model B1 + B2
│   │   └── explainer.py            ← Giải thích ngôn ngữ tự nhiên
│   └── pipeline/
│       ├── viral_system.py         ← Main class tổng hợp
│       ├── polling_monitor.py      ← Background polling video
│       └── report_generator.py    ← ChannelReport + VideoReport
├── trained_models/                 ← .pkl files sau training
├── tests/
│   ├── test_youtube_client.py
│   ├── test_feature_engineer.py
│   ├── test_models.py
│   └── test_pipeline.py
├── cache/                          ← API cache JSON + polling data
└── requirements.txt
```

---

## Kiến trúc Model

### Model A — Channel Viral Classifier
- **Algorithm**: Random Forest → Logistic Regression → SVM (tự chọn tốt nhất)
- **Training**: Leave-One-Out CV (phù hợp với 42 samples)
- **Metric**: F1-score + Accuracy

**Features**:
| Feature | Mô tả |
|---------|-------|
| f1_efficiency | total_views / subscribers |
| f2_loyalty | total_likes / total_views |
| f3_depth | total_comments / total_views |
| f4_consistency | 1 - CoV views |
| f6_avg_views | avg views/video |
| f7_engagement | loyalty×0.6 + depth×0.4 |
| f9_sub_tier | log10(subscribers) |
| f11_recent_trend | avg 5 video mới / avg 5 video cũ |
| cluster_id | K-Means cluster |

### Model B — Video Viral Prediction
- **B1**: Binary classifier (viral / not viral) — GBM / Logistic Regression
- **B2**: Multi-class time window ("viral_within_7d" / "viral_within_30d" / "not_viral")
- **Training**: Stratified K-Fold CV
- **Threshold**: P(viral) > 0.6 → chạy B2

### Channel Clustering
- **Algorithm**: K-Means (k tự tìm bằng Elbow + Silhouette)
- **Features**: log10(sub), log10(avg_views), loyalty_ratio, depth_ratio
- **Mục đích**: Tạo cluster prior probability cho kênh mới

---

## Label Creation (Bước 0)

### Channel Label
```
Label = 1 nếu ≥ 2/3 điều kiện:
  A: efficiency_ratio > p75
  B: loyalty_ratio > median
  C: avg_views_per_video > p75
```

### Video Label
```
relative_score = (views - channel_avg) / channel_std
Label = 1 nếu relative_score > 1.5
```

---

## API Quota (YouTube API v3)

| Operation | Units |
|-----------|-------|
| Search channel | 100 |
| Channel stats | 1 |
| Recent videos | 1 |
| Video stats (50 videos) | 1 |
| **Total per channel** | **~103** |

Quota giới hạn: 10,000 units/ngày → tối đa ~97 kênh/ngày.
Cache giúp tái sử dụng: search 6h, channel 24h, video 1h.

---

## Chạy Tests

```bash
# Từ root workspace
pytest ml/tests/ -v

# Test cụ thể
pytest ml/tests/test_models.py -v
pytest ml/tests/test_feature_engineer.py -v
```

---

## Lưu ý

- **Không commit** `.env` hoặc `credentials/*.json` lên git
- Trained models trong `ml/trained_models/` cũng nên thêm vào `.gitignore`
- Cache trong `ml/cache/` có thể xóa tự do (sẽ tự tạo lại)
- Với video mới đăng (< 6h): confidence = LOW, cần thêm data
- Với video > 48h: confidence = HIGH, kết quả đáng tin cậy nhất
