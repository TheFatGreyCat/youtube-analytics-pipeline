# BÁO CÁO DỰ ÁN: YouTube Analytics Pipeline

> **Ngày lập:** 03/03/2026  
> **Phiên bản:** 0.1.0  
> **Công nghệ chính:** Python 3.12 · Google BigQuery · dbt · Prefect · Streamlit · XGBoost

---

## MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Hạ tầng & Dịch vụ (Docker)](#4-hạ-tầng--dịch-vụ-docker)
5. [Module Extract – Thu thập dữ liệu](#5-module-extract--thu-thập-dữ-liệu)
6. [Module Orchestrate – Điều phối luồng](#6-module-orchestrate--điều-phối-luồng)
7. [Module dbt – Biến đổi dữ liệu](#7-module-dbt--biến-đổi-dữ-liệu)
8. [Module Serve – Dashboard trực quan](#8-module-serve--dashboard-trực-quan)
9. [Module ML – Dự đoán video Viral](#9-module-ml--dự-đoán-video-viral)
10. [Luồng xử lý dữ liệu end-to-end](#10-luồng-xử-lý-dữ-liệu-end-to-end)
11. [Quản lý Quota API YouTube](#11-quản-lý-quota-api-youtube)
12. [Chiến lược Crawl](#12-chiến-lược-crawl)
13. [Chất lượng dữ liệu & Testing](#13-chất-lượng-dữ-liệu--testing)
14. [Cấu hình & Biến môi trường](#14-cấu-hình--biến-môi-trường)
15. [Tổng kết & Hướng phát triển](#15-tổng-kết--hướng-phát-triển)

---

## 1. Tổng quan dự án

**YouTube Analytics Pipeline** là một hệ thống pipeline dữ liệu hoàn chỉnh (end-to-end) được xây dựng để tự động **thu thập, lưu trữ, biến đổi, trực quan hóa** và **dự đoán hiệu suất** của các kênh YouTube. Dự án hướng tới việc giúp người dùng theo dõi nhiều kênh YouTube cùng lúc, phân tích xu hướng nội dung, và dự đoán khả năng viral của video dựa trên Machine Learning.

### Mục tiêu chính

| Mục tiêu | Chi tiết |
|---|---|
| **Tự động hóa** | Tự động thu thập dữ liệu từ YouTube API theo lịch định sẵn |
| **Lưu trữ có cấu trúc** | Dữ liệu thô lưu BigQuery, metadata quản lý qua PostgreSQL |
| **Phân tích sâu** | dbt biến đổi dữ liệu thô thành các bảng phân tích có giá trị |
| **Trực quan hóa** | Dashboard Streamlit hiển thị KPI, biểu đồ, trending videos |
| **Dự đoán ML** | Mô hình XGBoost dự đoán video có khả năng viral hay không |

### Công nghệ sử dụng

```
Thu thập    : Python · YouTube Data API v3 · google-api-python-client
Lưu trữ     : PostgreSQL (metadata) · Google BigQuery (dữ liệu thô + analytics)
Biến đổi    : dbt-core · dbt-bigquery · dbt_utils · fivetran_utils
Điều phối   : Prefect 3.0 · Redis (message broker)
Dashboard   : Streamlit · Plotly · Pandas
ML          : XGBoost · Optuna · SHAP · scikit-learn
Hạ tầng     : Docker · Docker Compose
```

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        NGUỒN DỮ LIỆU                            │
│                   YouTube Data API v3                           │
│         (channels · videos · playlists · comments)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP / REST
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TẦNG THU THẬP (Extract)                       │
│  YouTubeCrawler  ──►  PostgresManager  (metadata, quota)        │
│       │          ──►  BigQueryManager  (raw JSON tables)        │
│       │                                                         │
│  crawl_channel()  · crawl_videos()  · crawl_comments()          │
│  crawl_channel_smart()  (full / incremental tự động)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┴──────────────────┐
          ▼                                   ▼
  ┌────────────────┐                 ┌──────────────────────┐
  │  PostgreSQL    │                 │    Google BigQuery   │
  │  (metadata)    │                 │   Dataset: raw_yt    │
  │                │                 │                      │
  │ channels_config│                 │ raw_channels         │
  │ crawl_log      │                 │ raw_videos           │
  │ api_quota      │                 │ raw_playlists        │
  └────────────────┘                 │ raw_comments         │
                                     └──────────┬───────────┘
                                                │
                           ┌────────────────────▼──────────────────┐
                           │         TẦNG BIẾN ĐỔI (dbt)           │
                           │                                       │
                           │  Staging  →  Intermediate  →  Mart    │
                           │                                       │
                           │  stg_youtube__videos                  │
                           │  stg_youtube__channels                │
                           │  stg_youtube__playlists               │
                           │        ↓                              │
                           │  int_videos__enhanced                 │
                           │  int_engagement_metrics               │
                           │  int_channel_summary                  │
                           │        ↓                              │
                           │  fct_video_performance                │
                           │  dim_channel_summary                  │
                           │  agg_daily_metrics                    │
                           └──────────┬────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────────┐
                    ▼                                       ▼
        ┌───────────────────────┐             ┌────────────────────────┐
        │   TẦNG DASHBOARD      │             │    TẦNG ML             │
        │     (Streamlit)       │             │  (XGBoost + Optuna)    │
        │                       │             │                        │
        │  Channel Search       │             │  data_loader.py        │
        │  KPI Cards            │             │  label.py              │
        │  Top Videos           │             │  features.py           │
        │  Trending Detection   │             │  train.py              │
        │  Engagement Charts    │             │  predict.py            │
        └───────────────────────┘             └────────────────────────┘
                    ▲
                    │
        ┌───────────────────────┐
        │  ĐIỀU PHỐI (Prefect)  │
        │                       │
        │  youtube_flow         │
        │  extract_flow         │
        │  dbt_flow             │
        │  monitoring_flow      │
        └───────────────────────┘
```

---

## 3. Cấu trúc thư mục

```
youtube-analytics-pipeline/
│
├── extract/               # Module thu thập dữ liệu
│   ├── cli.py             # CLI interface
│   ├── crawlers.py        # Logic crawl YouTube API
│   ├── db_manager.py      # PostgresManager + BigQueryManager
│   ├── config.py          # Đọc biến môi trường
│   ├── channel_finder.py  # Tìm kiếm kênh
│   └── schemas/           # SQL schema cho PostgreSQL & BigQuery
│
├── orchestrate/           # Module điều phối Prefect
│   ├── flows/
│   │   ├── youtube_flow.py     # Flow tổng hợp (extract + transform)
│   │   ├── extract_flow.py     # Flow thu thập
│   │   ├── dbt_flow.py         # Flow chạy dbt
│   │   ├── monitoring_flow.py  # Flow kiểm tra sức khỏe hệ thống
│   │   └── management_flow.py  # Flow quản lý kênh
│   ├── tasks/             # Prefect tasks
│   └── deployments/       # Cấu hình deployment
│
├── dbt_project/           # Module biến đổi dữ liệu
│   ├── models/
│   │   ├── sources.yml         # Khai báo nguồn dữ liệu BigQuery
│   │   ├── staging/            # Tầng staging (parse JSON thô)
│   │   ├── intermediate/       # Tầng trung gian (tính metrics)
│   │   └── mart/               # Tầng mart (bảng phân tích cuối)
│   ├── macros/            # Custom SQL macros
│   ├── tests/             # Data quality tests
│   └── seeds/             # Dữ liệu tĩnh (youtube_categories.csv)
│
├── serve/                 # Module dashboard
│   ├── app.py             # Trang chính phân tích kênh
│   ├── pages/
│   │   └── 1_Channel_Search.py  # Trang tìm kiếm kênh
│   └── requirements.txt
│
├── ml/                    # Module Machine Learning
│   ├── data_loader.py     # Load dữ liệu từ BigQuery
│   ├── label.py           # Định nghĩa nhãn is_viral
│   ├── features.py        # Feature engineering (6 nhóm, 40+ features)
│   ├── train.py           # Training pipeline (Optuna + XGBoost)
│   ├── predict.py         # Inference + SHAP explanation
│   └── save_load.py       # Lưu/load model
│
├── script/                # Utility scripts
│   ├── monitor_quota.py   # Giám sát quota API
│   ├── dbt_cli.py         # Wrapper chạy dbt
│   ├── deploy_prefect.py  # Deploy Prefect workflows
│   └── setup_all.py       # Script cài đặt tự động
│
├── config/
│   ├── channels.yml       # Danh sách kênh cần crawl
│   └── prefect.yaml       # Cấu hình lịch chạy Prefect
│
├── docker/                # Dockerfiles
├── compose.yml            # Docker Compose
├── pyproject.toml         # Python dependencies
└── Makefile               # Lệnh tiện ích
```

---

## 4. Hạ tầng & Dịch vụ (Docker)

Hệ thống được container hóa hoàn toàn bằng Docker Compose với 4 service chính:

### 4.1 Sơ đồ dịch vụ

```
┌──────────────────────────────────────────────────────────┐
│                     Docker Network                       │
│                                                          │
│  ┌─────────────┐    ┌──────────┐    ┌─────────────────┐  │
│  │  postgres   │    │  redis   │    │  prefect-server │  │
│  │  :5432      │    │  :6379   │    │  :4200          │  │
│  │             │    │          │    │                 │  │
│  │ Lưu metadata│    │ Message  │    │ Web UI +        │  │
│  │ crawl logs  │    │ Broker   │    │ API server      │  │
│  │ quota usage │    │ cho      │    │ PostgreSQL DB   │  │
│  └──────┬──────┘    │ Prefect  │    └────────┬────────┘  │
│         │           └──────────┘             │           │
│         └─────────────────────────────────────┤          │
│                                               │          │
│                          ┌────────────────────▼────────┐ │
│                          │      prefect-worker         │ │
│                          │                             │ │
│                          │  Chạy các Prefect flows:    │ │
│                          │  · extract (crawl YT)       │ │
│                          │  · dbt transform            │ │
│                          │  · monitoring               │ │
│                          └─────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### 4.2 Chi tiết từng service

| Service | Image | Port | Mục đích |
|---|---|---|---|
| `postgres` | postgres:14 | 5432 | Lưu metadata kênh, crawl log, quota |
| `redis` | redis:7 | 6379 | Message broker cho Prefect task queue |
| `prefect-server` | prefecthq/prefect:3-latest | 4200 | API server + Web UI quản lý flows |
| `prefect-worker` | Custom Dockerfile | — | Thực thi Prefect flows và tasks |

### 4.3 Volumes

- `postgres_data` — Dữ liệu PostgreSQL persistent
- `redis_data` — Cache Redis persistent
- `./credentials` → `/app/credentials:ro` — GCP service account key (read-only)
- `./config` → `/app/config:ro` — File cấu hình kênh
- `./logs` → `/app/logs` — Log output

---

## 5. Module Extract – Thu thập dữ liệu

### 5.1 Tổng quan

Module `extract/` chịu trách nhiệm kết nối YouTube Data API v3, thu thập dữ liệu thô và lưu vào 2 nơi song song:
- **PostgreSQL**: metadata quản lý (cấu hình kênh, lịch sử crawl, quota)
- **BigQuery**: raw JSON data (nội dung API response đầy đủ)

### 5.2 Class YouTubeCrawler

**File:** `extract/crawlers.py`

```
YouTubeCrawler
├── __init__()              → Build YouTube API client, khởi tạo PG + BQ manager
├── crawl_channel()         → Lấy thông tin tổng quan kênh (1 unit quota)
├── crawl_videos()          → Lấy tất cả video của kênh (full crawl)
├── crawl_videos_incremental() → Chỉ lấy video mới hơn lần crawl trước
├── crawl_playlists()       → Lấy danh sách playlist (1 unit quota)
├── crawl_comments()        → Lấy comments của video (1 unit/video)
├── crawl_channel_full()    → Chạy toàn bộ: channel + videos + playlists + comments
└── crawl_channel_smart()   → Tự quyết định full hay incremental dựa trên lịch sử
```

**Chi phí Quota theo từng operation:**

| API Operation | Quota Cost |
|---|---|
| `channels.list` | 1 unit |
| `videos.list` | 1 unit / batch 50 video |
| `playlistItems.list` | 1 unit / trang 50 mục |
| `playlists.list` | 1 unit |
| `commentThreads.list` | 1 unit / video |

**Ước tính tổng chi phí mỗi kênh:**
- Không có comments: **5–15 units**
- Có comments (10 video gần nhất): **15–25 units**
- Giới hạn an toàn hàng ngày: **10,000 units** → tối đa ~15 kênh/ngày

### 5.3 Class PostgresManager

**Mục đích:** Quản lý metadata, không lưu nội dung YouTube.

**Các bảng chính:**

```sql
channels_config        -- Cấu hình kênh cần crawl
    channel_id, channel_name, crawl_frequency_hours,
    last_crawl_ts, next_crawl_ts, crawl_status,
    last_video_published_at, priority, is_active

crawl_log              -- Lịch sử mỗi lần crawl
    channel_id, crawl_ts, records_fetched,
    status (success/failed), error_msg, execution_time_seconds

api_quota_usage        -- Theo dõi quota hàng ngày
    date, quota_used, daily_limit, updated_at
```

**Các hàm quan trọng:**

| Hàm | Mô tả |
|---|---|
| `get_channels_to_crawl(limit)` | Lấy danh sách kênh đến hạn crawl (next_crawl_ts ≤ NOW) |
| `should_full_crawl(channel_id)` | True nếu chưa crawl trong 7 ngày |
| `update_crawl_success()` | Cập nhật last_crawl_ts, tính next_crawl_ts |
| `get_api_quota_status()` | Trả về {quota_used, daily_limit, percentage_used} |
| `update_last_video_date()` | Lưu video mới nhất để incremental crawl |

### 5.4 Class BigQueryManager

**Mục đích:** Lưu raw JSON response từ YouTube API vào BigQuery.

**Các bảng BigQuery (dataset: `raw_yt`):**

| Bảng | Nội dung | Schema |
|---|---|---|
| `raw_channels` | Toàn bộ API response của channels.list | `id, raw (JSON), ingestion_time` |
| `raw_videos` | Toàn bộ API response của videos.list | `id, raw (JSON), ingestion_time` |
| `raw_playlists` | Toàn bộ API response của playlists.list | `id, channel_id, raw (JSON), ingestion_time` |
| `raw_comments` | Toàn bộ API response của commentThreads.list | `id, video_id, channel_id, raw (JSON), ingestion_time` |

**Cơ chế chống duplicate:** Trước khi insert, hệ thống kiểm tra `id` đã tồn tại chưa bằng BigQuery query → chỉ insert bản ghi mới.

**Phương thức insert:** Sử dụng `load_table_from_file()` (NEWLINE_DELIMITED_JSON) thay vì `insert_rows_json()` để tăng hiệu suất và độ tin cậy với batch lớn.

### 5.5 CLI Interface

```bash
python -m extract.cli setup                          # Tạo bảng DB
python -m extract.cli add UCxxx "Tên kênh"          # Thêm kênh
python -m extract.cli channels                       # Liệt kê kênh
python -m extract.cli crawl-file --limit 10          # Crawl từ channels.yml
python -m extract.cli crawl --channel UCxxx          # Crawl 1 kênh cụ thể
python -m extract.cli history --limit 20             # Xem lịch sử crawl
```

---

## 6. Module Orchestrate – Điều phối luồng

### 6.1 Tổng quan

Module `orchestrate/` sử dụng **Prefect 3.0** để tự động hóa và lên lịch toàn bộ pipeline. Các flows được deploy lên Prefect server và worker tự động thực thi theo lịch cron.

### 6.2 Sơ đồ Flows

```
youtube_analytics_pipeline  (Flow tổng)
│
├── system_health_check_flow  (monitoring_flow.py)
│   └── Kiểm tra: PostgreSQL, BigQuery, API quota, disk space
│
├── youtube_extract_flow  (extract_flow.py)
│   ├── get_channels_to_crawl_task()
│   ├── crawl_channel_task() × N kênh (parallel)
│   └── update_quota_task()
│
└── dbt_transformation_flow  (dbt_flow.py)
    ├── dbt deps
    ├── dbt run (staging → intermediate → mart)
    └── dbt test (data quality checks)
```

### 6.3 Các Deployment được cấu hình

| Deployment | Schedule | Mô tả |
|---|---|---|
| `daily-youtube-analytics` | 02:00 hàng ngày | Full pipeline: extract + dbt |
| `extract-3times-daily` | 08:00, 14:00, 20:00 | Chỉ crawl dữ liệu |
| `dbt-transform-daily` | 08:30, 14:30, 20:30 | Chỉ chạy dbt transform |

**Timezone:** Asia/Ho_Chi_Minh (UTC+7)

### 6.4 Cấu hình (config/prefect.yaml)

```yaml
schedule:
  cron: "0 2 * * *"
  timezone: Asia/Ho_Chi_Minh
parameters:
  crawl_limit: 10
  include_comments: false
  run_dbt_tests: true
```

---

## 7. Module dbt – Biến đổi dữ liệu

### 7.1 Tổng quan

Module `dbt_project/` sử dụng **dbt-bigquery** để biến đổi dữ liệu thô (raw JSON) trong BigQuery thành các bảng phân tích có cấu trúc theo kiến trúc 3 tầng.

### 7.2 Kiến trúc 3 tầng

```
BigQuery raw_yt               BigQuery analytics
(raw JSON tables)             (transformed tables)
                                      
raw_channels  ──┐             ┌── stg_youtube__channels
raw_videos    ──┤  STAGING    ├── stg_youtube__videos
raw_playlists ──┘  (views)   └── stg_youtube__playlists
                                       │
                              INTERMEDIATE (views)
                              ├── int_videos__enhanced
                              ├── int_engagement_metrics
                              └── int_channel_summary
                                       │
                              MART (tables, partitioned)
                              ├── fct_video_performance
                              ├── dim_channel_summary
                              └── agg_daily_metrics
```

### 7.3 Tầng Staging

Mục đích: **parse JSON thô** từ cột `raw` thành các cột có kiểu dữ liệu rõ ràng.

**`stg_youtube__videos`** (materialized: view)
- Dùng `JSON_VALUE()` và `JSON_QUERY()` của BigQuery để extract từng field
- Lọc: chỉ giữ video `public` và có `duration` hợp lệ
- Validate: join với `raw_channels` để loại bỏ video từ kênh không hợp lệ
- Cast kiểu: `viewCount`, `likeCount`, `commentCount` → INT64; `publishedAt` → TIMESTAMP
- Hỗ trợ passthrough columns (cấu hình qua macro)

**`stg_youtube__channels`** — Tương tự, extract subscriber_count, viewCount, country, created_at

**`stg_youtube__playlists`** — Extract title, itemCount, privacyStatus

### 7.4 Tầng Intermediate

Mục đích: **tính toán metrics** và **enrich dữ liệu** bằng cách join nhiều nguồn.

**`int_videos__enhanced`** (view)
- Join videos với channels để lấy: channel_name, subscriber_count, country_code
- Thêm cột phái sinh:
  - `duration_seconds` — macro `parse_iso8601_duration()` chuyển PT1H2M3S → giây
  - `published_date`, `published_year`, `published_month`, `published_dayofweek`, `published_hour`
  - `days_since_published` — DATE_DIFF(current_date, published_date, day)
  - `video_length_category` — shorts / short / medium / long

**`int_engagement_metrics`** (view)
- Tính toán các chỉ số tương tác:

```
like_rate_pct      = (like_count / view_count) × 100     [giới hạn max 100%]
comment_rate_pct   = (comment_count / view_count) × 100
engagement_score   = (like_count + comment_count × 2) / view_count × 100
avg_views_per_day  = view_count / days_since_published
engagement_level   = "high" nếu like_rate >= 5%, "medium" nếu >= 2%, "low" còn lại
is_potentially_viral = avg_views_per_day > channel_subscribers × 0.1
```

**`int_channel_summary`** (view)
- Tổng hợp thống kê toàn kênh: total_videos, total_views, avg_like_rate, avg_days_between_uploads...

### 7.5 Tầng Mart

Mục đích: **bảng phân tích cuối cùng** phục vụ dashboard và ML.

**`fct_video_performance`** (table, partitioned by published_date theo tháng, clustered by channel_id + video_id)

Chứa đầy đủ thông tin của mỗi video: metadata + engagement metrics + viral flag. Đây là bảng trung tâm của hệ thống.

**`dim_channel_summary`** (table)

Dimension table cho kênh, bổ sung thêm:
- `channel_active_days` — số ngày từ video đầu đến video mới nhất
- `views_per_subscriber` — tổng views / subscriber count
- `videos_per_week` — tần suất đăng video

**`agg_daily_metrics`** (table, partitioned by metric_date, clustered by channel_id + metric_date)

Bảng aggregate theo ngày:
- `videos_published`, `total_views`, `total_likes`, `total_comments`
- `avg_engagement_score`, `avg_like_rate_pct`, `max_video_views`
- Surrogate key: `date_channel_key` (dbt_utils.generate_surrogate_key)

### 7.6 Macros tùy chỉnh

| Macro | Mô tả |
|---|---|
| `parse_iso8601_duration(col)` | Chuyển PT1H2M3S thành số giây (BigQuery compatible) |
| `get_passthrough_columns(var)` | Cho phép thêm cột tùy chỉnh vào staging models |
| `generate_schema_name(schema, node)` | Tùy chỉnh tên schema theo môi trường |
| `deduplicate_by_latest(col)` | Dedup lấy bản ghi mới nhất |

### 7.7 Data Quality Tests

```sql
-- assert_positive_statistics.sql
-- Kiểm tra: view_count, like_count, comment_count >= 0

-- assert_valid_engagement_rates.sql  
-- Kiểm tra: like_rate_pct và comment_rate_pct trong [0, 100]
```

Ngoài ra mỗi model có inline tests trong YAML:
- `unique` và `not_null` trên primary keys
- `not_null` trên ingestion_time

---

## 8. Module Serve – Dashboard trực quan



## 9. Module ML – Dự đoán video Viral

### 9.1 Tổng quan

Module `ml/` xây dựng mô hình **phân loại nhị phân** dự đoán liệu một video có khả năng viral hay không, dựa trên dữ liệu từ tầng intermediate của dbt.

**Pipeline ML:**
```
data_loader.py → label.py → features.py → train.py → save_load.py
                                                           │
                                                      predict.py
```

### 9.2 Định nghĩa nhãn Viral (`label.py`)

**Chiến lược nhãn (`strategy`):**

| Strategy | Điều kiện viral | Ghi chú |
|---|---|---|
| `existing` | Dùng cột `is_potentially_viral` có sẵn từ dbt | avg_views_per_day > subs × 0.1 |
| `ratio` | `view_ratio >= 3.0x` (views / avg_channel_views) | Đơn giản, dễ giải thích |
| `combined` | `view_ratio >= 3.0x` HOẶC `velocity_score >= 1.0` | Kết hợp, recall cao hơn |
| `auto` | Tự chọn: existing nếu viral_rate 8–20%, ngược lại combined | Khuyến nghị |

**Auto-adjust threshold:** Nếu viral_rate ngoài [5%, 30%] sau khi apply, hệ thống tự điều chỉnh view_ratio_threshold (±0.5x mỗi lần, tối đa 5 lần) để đưa tỷ lệ về mức hợp lý.

**Validation nhãn (`validate_label_quality`):**
- Kiểm tra viral rate tổng thể (mục tiêu 8–20%)
- Phát hiện kênh bị bias (viral_rate = 0% hoặc 100%)
- Kiểm tra ổn định theo tháng (seasonal trend, cho phép lệch ≤ 15%)
- Kiểm tra data leakage (correlation với is_potentially_viral < 0.7)

### 9.3 Feature Engineering (`features.py`)

Hệ thống xây dựng **40 features** chia thành 6 nhóm:

**Nhóm A – Temporal (8 features)**
```
published_hour, published_dayofweek, published_month
is_weekend, is_prime_time (18-22h), is_morning (6-10h), is_lunch_slot (11-13h)
publish_quarter
```

**Nhóm B – Title (7 features)**
```
title_length, title_word_count
has_number, has_question, has_exclamation, has_emoji, has_caps_word
```

**Nhóm C – Content (11 features)**
```
duration_minutes, tag_count, has_tags, is_hd, has_caption
is_embeddable, is_made_for_kids
is_shorts, is_short, is_medium   (one-hot, is_long = reference)
category_id_enc                  (LabelEncoded)
```

**Nhóm D – Channel (8 features)**
```
subscriber_log               (log1p transform)
channel_age_days             (published_at - channel_created_at)
upload_freq_per_day          (1 / avg_days_between_uploads)
channel_avg_views_log, channel_avg_like_rate, channel_avg_cmt_rate
duration_vs_channel          (video_duration / channel_avg_duration)
total_videos_crawled
```

**Nhóm E – Engagement (5 features)**
```
avg_views_per_day_log        (log1p transform)
engagement_level_enc         (high=2, medium=1, low=0)
like_rate_pct, comment_rate_pct, engagement_score
```

**Nhóm F – Interaction (4 features)**
```
view_vs_channel_avg          (view_count / channel_avg_views)
like_rate_vs_channel         (like_rate / channel_avg_like_rate)
comment_rate_vs_channel      (comment_rate / channel_avg_cmt_rate)
velocity_score               (avg_views_per_day / (subscribers × 0.01))
```

**Fill missing values:**
- Binary features → điền 0
- Numeric features → điền median (tính từ training set, tái sử dụng khi predict)

### 9.4 Training Pipeline (`train.py`)

**Bước 1 – Time-based Split (không random)**

```
70% Train  │  15% Validation  │  15% Test
──────────────────────────────────────────►  thời gian
(sớm nhất)                       (mới nhất)
```
Sắp xếp theo `published_at` để tránh temporal data leakage.

**Bước 2 – Optuna Hyperparameter Tuning**
- Tối ưu hóa ROC-AUC trên validation set
- 50 trials mặc định, TPE sampler
- Search space: n_estimators (200–1500), max_depth (3–9), learning_rate (log-scale), subsample, colsample_bytree, min_child_weight, gamma, reg_alpha, reg_lambda
- `scale_pos_weight` cố định = n_negative / n_positive (xử lý class imbalance)

**Bước 3 – Train Final Model**
- Gộp train + validation (85% data) để train lại với best params
- Evaluate trên test set (15% cuối cùng theo thời gian)

**Bước 4 – Đánh giá Model**

Xuất ra 4 biểu đồ (lưu `plots/evaluation.png`):
1. ROC Curve (AUC)
2. Precision-Recall Curve + optimal threshold
3. Confusion Matrix
4. Feature Importance (Top 20, by Gain)

Gợi ý threshold:
- Ưu tiên recall (không bỏ sót viral) → threshold thấp hơn optimal
- Ưu tiên precision (chính xác khi predict viral) → threshold cao hơn

### 9.5 Prediction & Explanation (`predict.py`)

**Input:**
```python
video_data = {
    "video_id", "title", "published_at", "duration_iso8601",
    "view_count", "like_count", "comment_count", "tags",
    "category_id", "has_caption", "definition", ...
}
channel_data = {
    "channel_id", "channel_name", "subscriber_count",
    "channel_created_at", "country_code", ...
}
```

**Output:**
```python
{
    "viral_score":     0.7842,       # Xác suất viral [0, 1]
    "prediction":      "CO KHA NANG VIRAL",
    "confidence":      "Cao",        # Cao / Trung bình / Thấp
    "top_5_drivers": [               # SHAP-based explanations
        {"feature": "velocity_score", "impact": +0.234, "meaning": "Tốc độ tăng view..."},
        ...
    ],
    "recommendation":  "Video đang có dấu hiệu viral mạnh. Nen boost...",
    "warnings":        ["Video mới trong vòng 24h, kết quả có thể chưa ổn định"]
}
```

**Explanation bằng SHAP:**
- Dùng `shap.TreeExplainer` (không cần sampling, nhanh với XGBoost)
- Top 5 features có |SHAP value| lớn nhất → giải thích nguyên nhân dự đoán
- Mỗi feature có `meaning` bằng tiếng Việt để dễ hiểu

### 9.6 Model Artifacts

```
models/
├── xgb_viral_v1.json           # XGBoost model (JSON format)
├── xgb_viral_v1_config.pkl     # Feature config, medians, encoders, thresholds
└── feature_config_v1.yaml      # Human-readable feature configuration
```

---

## 10. Luồng xử lý dữ liệu end-to-end

### 10.1 Luồng đầy đủ

```
[1] CẤU HÌNH
    config/channels.yml  →  Danh sách kênh cần theo dõi
              │
              ▼
[2] THU THẬP (Extract)
    YouTubeCrawler.crawl_channel_smart()
    ├── Full crawl (lần đầu hoặc 7 ngày 1 lần)
    │   ├── crawl_channel()    → raw_channels (BQ)
    │   ├── crawl_videos()     → raw_videos (BQ)
    │   └── crawl_playlists()  → raw_playlists (BQ)
    └── Incremental crawl (những lần sau)
        ├── crawl_channel()    → cập nhật stats
        └── crawl_videos_incremental()  → chỉ video MỚI
    
    + Ghi vào PostgreSQL: crawl_log, update next_crawl_ts, update quota
              │
              ▼
[3] BIẾN ĐỔI (dbt)
    Staging → parse JSON, lọc, cast types
    Intermediate → join, tính metrics (like_rate, engagement_score, ...)
    Mart → bảng phân tích cuối (fct, dim, agg)
              │
              ▼
[4] SỬ DỤNG (Parallel)
    ├── Dashboard Streamlit  → query BigQuery, hiển thị KPI + charts
    └── ML Pipeline
        ├── data_loader → load từ int_videos__enhanced, int_engagement_metrics
        ├── label → định nghĩa is_viral
        ├── features → 40 features
        ├── train → XGBoost + Optuna
        └── predict → inference + SHAP explanation
```

## 11. Cấu hình & Biến môi trường

### 11.1 File .env (bắt buộc)

```env
# YouTube API
YOUTUBE_API_KEY=AIzaSy...

# Google Cloud
GCP_PROJECT_ID=your-project-id
BQ_DATASET_ID=raw_yt
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json

# PostgreSQL
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=youtube_analytics
PG_USER=postgres
PG_PASSWORD=your_password

# Prefect
PREFECT_SERVER_PORT=4200
```

### 11.2 config/channels.yml

```yaml
channels:
  - id: UCXuqSBlHAE6Xw-yeJA0Tunw
    name: Linus Tech Tips
    frequency_hours: 24   # Crawl mỗi 24 giờ
    priority: 1           # Ưu tiên cao hơn
    active: true
    include_comments: false
```

---

## 12. Tổng kết & Hướng phát triển

### 12.1 Những gì dự án đã đạt được

| Hạng mục | Trạng thái |
|---|---|
| Thu thập tự động đa kênh YouTube | ✅ Hoàn thành |
| Crawl thông minh (full / incremental) | ✅ Hoàn thành |
| Quản lý & bảo vệ quota API | ✅ Hoàn thành |
| Lưu trữ raw data trên BigQuery | ✅ Hoàn thành |
| Pipeline dbt 3 tầng (staging → mart) | ✅ Hoàn thành |
| Dashboard Streamlit real-time | ✅ Hoàn thành |
| Phát hiện trending video | ✅ Hoàn thành |
| Tự động hóa với Prefect + Docker | ✅ Hoàn thành |
| ML dự đoán viral (XGBoost + Optuna) | ✅ Hoàn thành |
| SHAP explainability cho ML | ✅ Hoàn thành |
| Data quality tests | ✅ Hoàn thành |

### 12.2 Điểm mạnh kỹ thuật

- **Kiến trúc phân tách rõ ràng**: Mỗi tầng (extract, transform, serve, ml) độc lập, dễ mở rộng
- **Idempotent**: Có thể chạy lại mà không tạo duplicate (BigQuery dedup, dbt incremental)
- **Quan sát được (Observable)**: Prefect UI, quota monitoring, crawl log đầy đủ
- **Anti data leakage**: Time-based split trong ML, tách biệt train medians cho prediction
- **Giải thích được (Explainable AI)**: SHAP values giải thích từng dự đoán bằng tiếng Việt

### 15.3 Hướng phát triển

| Ưu tiên | Tính năng |
|---|---|
| Cao | Tích hợp kết quả ML vào dashboard Streamlit |
| Cao | Thêm trang so sánh nhiều kênh cùng lúc |
| Trung bình | Cảnh báo email/Slack khi video trending |
| Trung bình | Sentiment analysis cho comments |
| Thấp | Dự đoán số lượt view sau 7/30 ngày (regression) |
| Thấp | Hỗ trợ YouTube Shorts analytics riêng biệt |
| Thấp | API REST để tích hợp với hệ thống ngoài |

---

*Báo cáo được tạo tự động từ source code – YouTube Analytics Pipeline v0.1.0*
