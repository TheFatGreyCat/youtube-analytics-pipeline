# Hướng Dẫn Vận Hành Hệ Thống YouTube Analytics Pipeline

> **Phiên bản:** 1.0.0 | **Cập nhật:** 2026-03-03  
> **Stack:** Python 3.12 · PostgreSQL · BigQuery · dbt · Prefect · Streamlit · Docker

---

## Mục Lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cài đặt lần đầu](#3-cài-đặt-lần-đầu)
4. [Cấu hình credentials](#4-cấu-hình-credentials)
5. [Khởi động dịch vụ](#5-khởi-động-dịch-vụ)
6. [Quản lý kênh YouTube](#6-quản-lý-kênh-youtube)
7. [Thu thập dữ liệu (Extract)](#7-thu-thập-dữ-liệu-extract)
8. [Biến đổi dữ liệu (dbt Transform)](#8-biến-đổi-dữ-liệu-dbt-transform)
9. [Tự động hóa với Prefect](#9-tự-động-hóa-với-prefect)
10. [Xem kết quả (Dashboard)](#10-xem-kết-quả-dashboard)
11. [Giám sát & quota API](#11-giám-sát--quota-api)
12. [Quy trình vận hành hàng ngày](#12-quy-trình-vận-hành-hàng-ngày)
13. [Xử lý sự cố](#13-xử-lý-sự-cố)
14. [Tham khảo lệnh nhanh](#14-tham-khảo-lệnh-nhanh)

---

## 1. Tổng Quan Kiến Trúc

```
YouTube API
    │
    ▼
[Extract Layer]          Python · extract/
    │  crawl videos, channels, playlists
    ▼
[PostgreSQL]             Raw staging store (Docker)
    │
    ▼
[BigQuery]               Cloud data warehouse (GCP)
    │
    ▼
[dbt Transform Layer]    dbt_project/models/
    │  staging → intermediate → mart
    ▼
[Mart Tables]            fct_video_performance
    │                    dim_channel_summary
    │                    agg_daily_metrics
    ▼
[Streamlit Dashboard]    serve/app.py
```

**Luồng dữ liệu:**
```
channels.yml ──► crawl ──► PostgreSQL (raw) ──► BigQuery (raw)
                                                      │
                                              dbt staging (views)
                                                      │
                                           dbt intermediate (views)
                                                      │
                                              dbt mart (tables)
                                                      │
                                           Streamlit Dashboard
```

---

## 2. Yêu Cầu Hệ Thống

### Phần mềm bắt buộc

| Phần mềm | Phiên bản tối thiểu | Kiểm tra |
|---|---|---|
| Python | 3.12+ | `python --version` |
| Docker Desktop | 24+ | `docker --version` |
| Git | 2.x | `git --version` |
| Make (tùy chọn) | any | `make --version` |

### Tài khoản & dịch vụ cloud

| Dịch vụ | Mục đích | Ghi chú |
|---|---|---|
| Google Cloud Platform | BigQuery data warehouse | Cần project ID |
| YouTube Data API v3 | Thu thập dữ liệu kênh | Quota 10.000 units/ngày |
| GCP Service Account | Xác thực BigQuery | File JSON key |

### Cấp quyền Service Account (tối thiểu)

```
BigQuery Data Editor
BigQuery Job User
BigQuery Read Session User
```

---

## 3. Cài Đặt Lần Đầu

### Bước 3.1 — Clone repository và tạo môi trường ảo

```bash
git clone <repo-url>
cd youtube-analytics-pipeline

# Tạo virtualenv
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source .venv/bin/activate
```

### Bước 3.2 — Cài đặt dependencies

```bash
# Cài đầy đủ tất cả components
pip install -e ".[all]"
```

Các nhóm package:
| Option | Bao gồm |
|---|---|
| `[extract]` | psycopg2, isodate — thu thập dữ liệu |
| `[orchestrate]` | prefect, dbt-bigquery — tự động hóa & transform |
| `[serve]` | streamlit, plotly — dashboard |
| `[dev]` | pytest, jupyter — phát triển |
| `[all]` | Tất cả ở trên |

### Bước 3.3 — Tạo file .env

Tạo file `.env` tại thư mục gốc:

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

> ⚠️ File `.env` đã được gitignore — **không commit** file này.

---

## 4. Cấu Hình Credentials

### Bước 4.1 — Điền thông tin vào file .env

Mở file `.env` và cập nhật các giá trị sau:

```dotenv
# ── YouTube API ──────────────────────────────────────
YOUTUBE_API_KEY=AIza...            # Lấy từ Google Cloud Console > APIs & Services

# ── Google Cloud Platform ────────────────────────────
GCP_PROJECT_ID=my-project-123      # Project ID trên GCP Console
BQ_DATASET_ID=raw_yt               # Dataset BigQuery lưu raw data

# ── Service Account ──────────────────────────────────
GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json

# ── PostgreSQL (Docker) ──────────────────────────────
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=youtube_analytics
PG_USER=postgres
PG_PASSWORD=your_secure_password   # Đặt mật khẩu mạnh

# ── Prefect ──────────────────────────────────────────
PREFECT_SERVER_PORT=4200
```

### Bước 4.2 — Đặt file Service Account

```bash
# Đặt file JSON key vào thư mục credentials/
credentials/
└── service-account.json     ← đặt file key ở đây
```

Thư mục `credentials/` đã được gitignore và được mount vào Docker container dưới dạng read-only.

### Bước 4.3 — Lấy YouTube API Key

1. Truy cập [Google Cloud Console](https://console.cloud.google.com)
2. Chọn project → **APIs & Services** → **Credentials**
3. **Create Credentials** → **API Key**
4. Giới hạn key chỉ cho **YouTube Data API v3**
5. Copy key vào `YOUTUBE_API_KEY` trong `.env`

---

## 5. Khởi Động Dịch Vụ

### Bước 5.1 — Khởi động Docker services

```bash
# Khởi động PostgreSQL + Redis + Prefect server + worker
docker compose up -d

# Hoặc dùng Make
make up
```

Kiểm tra trạng thái:

```bash
docker compose ps
```

Kết quả mong đợi:
```
NAME               STATUS          PORTS
postgres           healthy         0.0.0.0:5432->5432/tcp
redis              healthy         6379/tcp
prefect-server     healthy         0.0.0.0:4200->4200/tcp
prefect-worker     running
```

> ⏳ Lần đầu khởi động mất khoảng **2–3 phút** để Prefect server sẵn sàng.

### Bước 5.2 — Khởi tạo database

Tạo các bảng cần thiết trong PostgreSQL và BigQuery:

```bash
python -m extract.cli setup

# Hoặc dùng Make
make setup
```

Lệnh này tạo:
- Bảng `channels_config` và `crawl_logs` trong **PostgreSQL**
- Dataset và bảng raw trong **BigQuery**

### Bước 5.3 — Cài đặt dbt packages

```bash
cd dbt_project
dbt deps

# Hoặc từ thư mục gốc
python script/dbt_cli.py deps
```

### Bước 5.4 — Kiểm tra kết nối dbt → BigQuery

```bash
python script/dbt_cli.py debug
```

Output thành công:
```
All checks passed!
  Connection: OK
  Required dependencies: OK
```

---

## 6. Quản Lý Kênh YouTube

### 6.1 Thêm kênh — cách 1: theo tên (tự động tìm ID)

```bash
python -m extract.cli add-by-name "@MrBeast"
python -m extract.cli add-by-name "Linus Tech Tips" --priority 3 --frequency 12
```

Tham số:
- `--priority`: 1 (thấp) đến 5 (cao), mặc định tự tính theo lượt sub
- `--frequency`: tần suất crawl tính bằng giờ, mặc định tự tính

### 6.2 Thêm kênh — cách 2: theo Channel ID

```bash
python -m extract.cli add UCXuqSBlHAE6Xw-yeJA0Tunw "Linus Tech Tips"
python -m extract.cli add UC-lHJZR3Gqxm24_Vd_AJ5Yw  "PewDiePie" --frequency 24
```

> Channel ID luôn bắt đầu bằng `UC` và dài 24 ký tự.

### 6.3 Thêm nhiều kênh — nhập từ CSV

```bash
# Dùng file template có sẵn
python -m extract.cli discover config/channels_template.csv \
    --output config/channels_found.csv

# Hoặc Make
make discover
```

Định dạng file CSV input:
```csv
channel_name
MrBeast
Linus Tech Tips
MKBHD
```

### 6.4 Xem danh sách kênh

```bash
# Kênh trong database (PostgreSQL)
python -m extract.cli list
make list

# Kênh trong file config (channels.yml)
python -m extract.cli channels
```

### 6.5 Xóa kênh

```bash
python -m extract.cli remove UCXuqSBlHAE6Xw-yeJA0Tunw
make remove-channel ID=UCXuqSBlHAE6Xw-yeJA0Tunw
```

### 6.6 Cấu hình thủ công qua channels.yml

Mở file `config/channels.yml` để thêm kênh và điều chỉnh cài đặt toàn cục:

```yaml
settings:
  max_videos_per_channel: 200      # Số video tối đa mỗi kênh
  max_comments_per_video: 100      # Số comment tối đa mỗi video
  batch_size: 100
  retry_failed_after_hours: 1
  api_delay_seconds: 0.5           # Độ trễ giữa các API call (giây)
  active: true
  frequency_hours: 24

channels:
  - id: UCXuqSBlHAE6Xw-yeJA0Tunw
    name: Linus Tech Tips
    frequency_hours: 12            # Crawl mỗi 12 giờ
    priority: 4
    active: true
    include_comments: false

  - id: UC-lHJZR3Gqxm24_Vd_AJ5Yw
    name: PewDiePie
    frequency_hours: 24
    priority: 3
    active: true
    include_comments: false
```

---

## 7. Thu Thập Dữ Liệu (Extract)

### 7.1 Kiểm tra quota trước khi crawl

**Luôn kiểm tra quota trước** để tránh vượt giới hạn 10.000 units/ngày:

```bash
python -m extract.cli quota
make quota
```

Ngưỡng cảnh báo:
| Mức sử dụng | Trạng thái | Hành động |
|---|---|---|
| < 50% | 🟢 Tốt | Crawl bình thường |
| 50–80% | 🟡 Thận trọng | Giảm số kênh |
| 80–90% | 🟠 Cảnh báo | Chỉ crawl kênh ưu tiên cao |
| > 90% | 🔴 Nguy hiểm | Hệ thống tự dừng |

Chi phí quota ước tính:
```
Mỗi kênh (không có comment):  ~5–15 units
Mỗi kênh (có comment):        ~15–25 units
Khuyến nghị an toàn:          ≤ 15 kênh/ngày không comment
                              ≤ 10 kênh/ngày có comment
```

### 7.2 Các chế độ crawl

#### Smart crawl (khuyến nghị — tự động chọn full/incremental)

```bash
python -m extract.cli crawl-smart --limit 20
make crawl-smart
```

Tự động quyết định:
- **Full crawl**: kênh chưa từng crawl hoặc đã crawl > 7 ngày
- **Incremental crawl**: chỉ lấy video mới kể từ lần crawl cuối

#### Crawl kênh chưa từng được crawl

```bash
python -m extract.cli crawl-new --limit 10
make crawl-new
```

#### Crawl theo lịch (đã đến hạn)

```bash
python -m extract.cli crawl-scheduled --limit 10
make crawl
```

#### Crawl từ file channels.yml

```bash
python -m extract.cli crawl-file --limit 15
```

#### Crawl một kênh cụ thể

```bash
# Không có comment
python -m extract.cli crawl --channel UCXuqSBlHAE6Xw-yeJA0Tunw

# Có comment (tốn thêm quota)
python -m extract.cli crawl --channel UCXuqSBlHAE6Xw-yeJA0Tunw --with-comments
```

### 7.3 Xem lịch sử crawl

```bash
# 20 bản ghi gần nhất
python -m extract.cli history --limit 20
make history

# Lịch sử của một kênh cụ thể
python -m extract.cli history --channel UCXuqSBlHAE6Xw-yeJA0Tunw
```

---

## 8. Biến Đổi Dữ Liệu (dbt Transform)

### 8.1 Sơ đồ các lớp dbt

```
BigQuery raw tables
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  STAGING (views)           schema: <dataset>_staging │
│  stg_youtube__videos                                 │
│  stg_youtube__channels                               │
│  stg_youtube__playlists                              │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  INTERMEDIATE (views)   schema: <dataset>_intermediate│
│  int_videos__enhanced     ← videos + channel context │
│  int_engagement_metrics   ← tính toán engagement    │
│  int_channel_summary      ← tổng hợp theo kênh      │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  MART (tables)             schema: <dataset>_mart    │
│  fct_video_performance    ← fact table video         │
│  dim_channel_summary      ← dimension table kênh    │
│  agg_daily_metrics        ← aggregate theo ngày     │
└─────────────────────────────────────────────────────┘
```

### 8.2 Chạy toàn bộ pipeline (khuyến nghị)

```bash
python script/dbt_cli.py pipeline
make dbt-pipeline
```

Gồm 3 bước tuần tự:
1. `dbt deps` — cài đặt/cập nhật packages
2. `dbt run` — build tất cả models
3. `dbt test` — kiểm tra chất lượng dữ liệu

### 8.3 Chạy từng lớp riêng lẻ

```bash
# Chỉ staging
python script/dbt_cli.py run --select staging.*
make dbt-staging

# Chỉ intermediate
python script/dbt_cli.py run --select intermediate.*

# Chỉ mart
python script/dbt_cli.py run --select mart.*
make dbt-mart

# Một model cụ thể
python script/dbt_cli.py run --select fct_video_performance
```

### 8.4 Chạy tests

```bash
# Tất cả tests
python script/dbt_cli.py test
make dbt-test

# Test một model cụ thể
python script/dbt_cli.py test --select fct_video_performance

# Test một lớp cụ thể
python script/dbt_cli.py test --select mart.*
```

### 8.5 Full refresh (rebuild toàn bộ bảng)

Dùng khi cần rebuild lại từ đầu, ví dụ sau khi sửa logic:

```bash
python script/dbt_cli.py full-refresh
```

### 8.6 Target environments

| Target | Dataset BigQuery | Dùng khi |
|---|---|---|
| `dev` (mặc định) | `BQ_DATASET_ID` từ `.env` | Phát triển, test |
| `prod` | `yt_dbt` (cố định) | Production |

```bash
# Chạy trên prod
python script/dbt_cli.py pipeline --target prod
```

---

## 9. Tự Động Hóa Với Prefect

### 9.1 Deploy workflows

```bash
python script/deploy_prefect.py
make prefect-deploy
```

Ba workflow được deploy:

| Tên | Lịch chạy (GMT+7) | Mô tả |
|---|---|---|
| `daily-youtube-analytics` | 02:00 hàng ngày | Pipeline đầy đủ (crawl + dbt) |
| `extract-3times-daily` | 08:00 · 14:00 · 20:00 | Chỉ crawl dữ liệu |
| `dbt-transform-daily` | 08:30 · 14:30 · 20:30 | Chỉ dbt transform |

### 9.2 Truy cập Prefect UI

```
http://localhost:4200
```

Tại đây có thể:
- Xem lịch sử các flow run
- Theo dõi trạng thái realtime
- Trigger chạy thủ công
- Xem logs chi tiết từng step

### 9.3 Chạy workflow thủ công (không chờ lịch)

```bash
docker compose exec prefect-worker \
    prefect deployment run youtube-analytics-pipeline/daily-youtube-analytics
```

### 9.4 Tùy chỉnh lịch chạy

Sửa file `config/prefect.yaml`:

```yaml
# Ví dụ: chạy lúc 4 giờ sáng
schedule:
  cron: "0 4 * * *"
  timezone: Asia/Ho_Chi_Minh

# Ví dụ: chạy mỗi 6 giờ
schedule:
  cron: "0 */6 * * *"
  timezone: Asia/Ho_Chi_Minh

# Ví dụ: chỉ chạy ngày thường
schedule:
  cron: "0 2 * * 1-5"
  timezone: Asia/Ho_Chi_Minh
```

Sau khi sửa, redeploy:

```bash
python script/deploy_prefect.py
```

### 9.5 Quản lý Docker services

```bash
# Xem log worker realtime
docker compose logs -f prefect-worker
make prefect-logs

# Xem log tất cả services
docker compose logs -f
make logs

# Restart worker (sau khi sửa code)
docker compose restart prefect-worker
make prefect-restart

# Rebuild worker image (sau khi thêm dependency)
docker compose up -d --build prefect-worker

# Dừng tất cả
docker compose down
make down

# Xóa sạch kể cả volumes (⚠️ mất toàn bộ data)
docker compose down -v
make clean
```

---

## 10. Xem Kết Quả (Dashboard)

### 10.1 Khởi động Streamlit

```bash
streamlit run serve/app.py
```

Truy cập tại: `http://localhost:8501`

### 10.2 Các bảng mart có thể query trực tiếp trên BigQuery

| Bảng | Mô tả | Partition | Cluster |
|---|---|---|---|
| `fct_video_performance` | Hiệu suất từng video | `published_date` (month) | `channel_id`, `video_id` |
| `dim_channel_summary` | Tổng hợp kênh | — | `channel_id` |
| `agg_daily_metrics` | Aggregate theo ngày | `metric_date` (month) | `channel_id`, `metric_date` |

Ví dụ query top 10 video engagement cao nhất:

```sql
SELECT
    title,
    channel_name,
    view_count,
    engagement_score,
    engagement_level
FROM `<project>.<dataset>_mart.fct_video_performance`
WHERE published_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
ORDER BY engagement_score DESC
LIMIT 10;
```

---

## 11. Giám Sát & Quota API

### 11.1 Kiểm tra toàn diện hệ thống

```bash
python script/monitor_quota.py
make monitor
```

Hiển thị:
- Quota đã dùng / còn lại
- Danh sách kênh và lần crawl cuối
- Trạng thái sức khỏe hệ thống

### 11.2 Các lệnh monitor chuyên biệt

```bash
# Chỉ xem quota
python script/monitor_quota.py --quota-only
make quota

# Chỉ xem danh sách kênh
python script/monitor_quota.py --channels-only

# Ước tính chi phí trước khi crawl N kênh
python script/monitor_quota.py --estimate 15
```

### 11.3 Xem log dbt

```bash
# Log file chi tiết
cat dbt_project/logs/dbt.log

# Query log (SQL đã chạy)
cat dbt_project/logs/query_log.sql
```

---

## 12. Quy Trình Vận Hành Hàng Ngày

### Chế độ thủ công (không dùng Prefect)

```bash
# 1. Kiểm tra quota
python -m extract.cli quota

# 2. Crawl dữ liệu
python -m extract.cli crawl-smart --limit 15

# 3. Transform với dbt
python script/dbt_cli.py pipeline

# 4. Kiểm tra kết quả
python script/monitor_quota.py
```

### Chế độ tự động (dùng Prefect — khuyến nghị)

Sau khi deploy Prefect workflows, hệ thống sẽ tự vận hành:

```
02:00  ──► daily-youtube-analytics   (crawl 15 kênh + dbt pipeline)
08:00  ──► extract-3times-daily      (crawl 10 kênh)
08:30  ──► dbt-transform-daily       (dbt run + test)
14:00  ──► extract-3times-daily
14:30  ──► dbt-transform-daily
20:00  ──► extract-3times-daily
20:30  ──► dbt-transform-daily
```

Việc cần làm hàng ngày:
1. **Sáng**: Kiểm tra Prefect UI xem các flow có chạy thành công không
2. **Nếu có lỗi**: Xem log và xử lý theo [mục 13](#13-xử-lý-sự-cố)

---

## 13. Xử Lý Sự Cố

### ❌ Lỗi: Quota vượt giới hạn

```
QuotaExceededError: Daily quota limit reached
```

**Nguyên nhân**: Vượt 10.000 units/ngày  
**Xử lý**:
```bash
# Xem trạng thái quota
python -m extract.cli quota

# Đợi đến 00:00 giờ Pacific (07:00 GMT+7) để quota reset
# Hoặc giảm số kênh crawl trong config/prefect.yaml
```

---

### ❌ Lỗi: Không kết nối được PostgreSQL

```
psycopg2.OperationalError: could not connect to server
```

**Nguyên nhân**: Docker chưa chạy hoặc sai credentials  
**Xử lý**:
```bash
# Kiểm tra trạng thái Docker
docker compose ps

# Khởi động lại nếu chưa chạy
docker compose up -d

# Kiểm tra log postgres
docker compose logs postgres
```

---

### ❌ Lỗi: BigQuery permission denied

```
google.api_core.exceptions.Forbidden: 403 Access Denied
```

**Nguyên nhân**: Service account thiếu quyền hoặc sai đường dẫn key file  
**Xử lý**:
1. Kiểm tra file key tồn tại: `ls credentials/`
2. Kiểm tra đường dẫn trong `.env`: `GOOGLE_APPLICATION_CREDENTIALS=credentials/service-account.json`
3. Kiểm tra quyền trong GCP Console: Service account cần `BigQuery Data Editor` + `BigQuery Job User`
4. Test kết nối: `python script/dbt_cli.py debug`

---

### ❌ Lỗi: Channel not found

```
ChannelNotFoundError: Channel 'XYZ' not found
```

**Nguyên nhân**: Tên kênh không chính xác hoặc kênh đã bị xóa  
**Xử lý**:
```bash
# Thử dùng @handle thay vì tên
python -m extract.cli add-by-name "@ChannelHandle"

# Kiểm tra Channel ID thủ công trên YouTube rồi thêm bằng ID
python -m extract.cli add UC... "Channel Name"
```

---

### ❌ Lỗi: dbt model failure

```
Compilation Error: ...
```

**Xử lý**:
```bash
# Xem chi tiết lỗi
python script/dbt_cli.py run --select <model_name>

# Kiểm tra log
cat dbt_project/logs/dbt.log

# Compile để kiểm tra SQL
python script/dbt_cli.py compile --select <model_name>
```

---

### ❌ Lỗi: Prefect worker không nhận job

**Xử lý**:
```bash
# Xem log worker
docker compose logs prefect-worker

# Restart worker
docker compose restart prefect-worker

# Kiểm tra work pool
docker compose exec prefect-worker prefect work-pool ls
```

---

### ❌ Lỗi: dbt test thất bại

```
FAIL [not_null] fct_video_performance.view_count
```

**Xử lý**:
```bash
# Xem chi tiết test failure trong BigQuery
# dbt lưu kết quả fail tại dataset: <dataset>_test_failures

# Chạy lại pipeline với full-refresh để rebuild
python script/dbt_cli.py full-refresh
```

---

## 14. Tham Khảo Lệnh Nhanh

### Docker

```bash
make up                          # Khởi động tất cả services
make down                        # Dừng tất cả services
make logs                        # Xem logs realtime
make clean                       # Xóa sạch kể cả volumes (⚠️)
docker compose ps                # Kiểm tra trạng thái
docker compose restart prefect-worker  # Restart worker
```

### Quản lý kênh

```bash
make list                                          # Danh sách kênh trong DB
python -m extract.cli channels                     # Danh sách trong channels.yml
python -m extract.cli add-by-name "@Handle"        # Thêm kênh theo tên
python -m extract.cli add UC... "Name"             # Thêm kênh theo ID
python -m extract.cli remove UC...                 # Xóa kênh
make discover                                      # Tìm kênh từ CSV
```

### Thu thập dữ liệu

```bash
make quota                                         # Kiểm tra quota
make crawl-smart                                   # Smart crawl (20 kênh)
make crawl-new                                     # Crawl kênh mới (10 kênh)
make crawl                                         # Crawl theo lịch (10 kênh)
make history                                       # Xem lịch sử crawl
python -m extract.cli crawl --channel UC...        # Crawl 1 kênh cụ thể
```

### dbt

```bash
make dbt-pipeline                                  # Full pipeline (deps+run+test)
make dbt-run                                       # Chỉ run models
make dbt-test                                      # Chỉ run tests
make dbt-staging                                   # Chỉ staging layer
make dbt-mart                                      # Chỉ mart layer
python script/dbt_cli.py full-refresh              # Rebuild toàn bộ
python script/dbt_cli.py debug                     # Kiểm tra kết nối
```

### Prefect

```bash
make prefect-deploy                                # Deploy/update workflows
make prefect-logs                                  # Xem log worker
make prefect-restart                               # Restart worker
# Truy cập UI: http://localhost:4200
```

### Giám sát

```bash
make monitor                                       # Toàn diện hệ thống
make quota                                         # Chỉ quota
python script/monitor_quota.py --estimate 15       # Ước tính chi phí
```

---

> **Lưu ý quan trọng:**
> - Quota YouTube API reset lúc **07:00 sáng (GMT+7)** mỗi ngày
> - Không commit file `.env` và `credentials/` lên git
> - Luôn kiểm tra quota trước khi crawl thủ công
> - Dùng `docker compose down -v` sẽ **xóa toàn bộ dữ liệu** trong PostgreSQL
