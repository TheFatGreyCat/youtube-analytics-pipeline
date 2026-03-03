import streamlit as st
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

# Thêm thư mục gốc vào path để import các module ML
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

# Nạp biến môi trường
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Import các hàm ML kèm xử lý lỗi
try:
    from ml.predict import predict_from_api_data
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    IMPORT_ERROR = str(e)

# ================= CẤU HÌNH =================
API_KEY = os.getenv('YOUTUBE_API_KEY') or ""
if not API_KEY:
    st.error("🔑 YOUTUBE_API_KEY chưa được thiết lập. Thêm vào .env và khởi động lại.")
    st.stop()

# Khởi tạo YouTube client
try:
    youtube = build("youtube", "v3", developerKey=API_KEY)
except Exception as err:
    st.error(f"❌ Không thể khởi tạo YouTube client: {err}")
    st.stop()

st.set_page_config(page_title="ML Viral Prediction", page_icon="🤖", layout="wide")

# Kiểm tra module ML có sẵn hay không
if not ML_AVAILABLE:
    st.error("❌ Không thể load mô hình ML!")
    st.markdown(f"**Lỗi:** `{IMPORT_ERROR}`")
    
    st.markdown("---")
    st.subheader("🔧 Cách Khắc Phục")
    
    st.markdown("""
    ### Bước 1️⃣: Cài đặt thư viện bị thiếu
    
    Mở terminal và chạy:
    ```bash
    .venv\\Scripts\\pip install shap xgboost optuna scikit-learn
    ```
    
    Hoặc cài tất cả dependencies:
    ```bash
    pip install -r ml/requirements.txt
    ```
    
    ### Bước 2️⃣: **Khởi động lại Streamlit** (QUAN TRỌNG!)
    
    Sau khi cài đặt xong, bạn **PHẢI khởi động lại Streamlit**:
    
    1. Vào terminal đang chạy Streamlit
    2. Nhấn `Ctrl + C` để dừng
    3. Chạy lại: `streamlit run serve/app.py`
    
    ### Bước 3️⃣: Tải lại trang này
    
    Sau khi khởi động lại Streamlit, tải lại trang này (F5 hoặc bấm vào "ML Prediction" ở sidebar).
    """)
    
    st.warning("⚠️ **Lưu ý:** Chỉ khởi động lại Streamlit trong terminal, không phải chỉ tải lại trang web!")
    
    st.info("💡 Nếu vẫn gặp lỗi, kiểm tra xem bạn đang dùng đúng Python environment (.venv) không.")
    
    st.stop()

# ================= CÁC HÀM =================

def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

def extract_video_id(url_or_id: str) -> str:
    """Tách video ID từ URL YouTube hoặc trả về trực tiếp nếu đã là ID."""
    url_or_id = url_or_id.strip()
    
    # Nếu đã là ID (11 ký tự)
    if len(url_or_id) == 11 and '/' not in url_or_id:
        return url_or_id
    
    # Tách từ nhiều định dạng URL khác nhau
    if 'youtu.be/' in url_or_id:
        return url_or_id.split('youtu.be/')[-1].split('?')[0]
    elif 'watch?v=' in url_or_id:
        return url_or_id.split('watch?v=')[-1].split('&')[0]
    elif 'youtube.com/embed/' in url_or_id:
        return url_or_id.split('embed/')[-1].split('?')[0]
    
    return url_or_id

def get_video_data(video_id: str) -> dict:
    """Lấy dữ liệu video từ YouTube API."""
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        
        if not response["items"]:
            return None
        
        item = response["items"][0]
        snippet = item["snippet"]
        content = item["contentDetails"]
        stats = item["statistics"]
        
        return {
            "video_id": video_id,
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "tags": snippet.get("tags", []),
            "category_id": snippet.get("categoryId", ""),
            "default_language": snippet.get("defaultLanguage", ""),
            "published_at": snippet.get("publishedAt", ""),
            "duration_iso8601": content.get("duration", ""),
            "has_caption": content.get("caption") == "true",
            "definition": content.get("definition", ""),
            "is_embeddable": content.get("embeddable", True),
            "is_made_for_kids": item.get("status", {}).get("madeForKids", False),
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)) if stats.get("likeCount") else None,
            "comment_count": int(stats.get("commentCount", 0)) if stats.get("commentCount") else 0,
            "channel_id": snippet.get("channelId", ""),
        }
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            st.error("🚫 **YouTube API Quota đã hết!**")
            st.warning("""
            **Nguyên nhân:** API key đã vượt quá giới hạn 10,000 units/ngày.
            
            **Giải pháp:**
            1. ⏰ Chờ đến ngày mai (quota tự động reset lúc 0:00 UTC)
            2. 🔑 Sử dụng API key khác
            3. 💰 Nâng cấp quota trong Google Cloud Console
            
            **Chi tiết:**
            - Free tier: 10,000 units/day
            - Mỗi video request: ~3-5 units
            - Mỗi search request: ~100 units
            """)
        else:
            st.error(f"❌ Lỗi khi lấy dữ liệu video: {str(e)}")
        return None

def get_channel_data(channel_id: str) -> dict:
    """Lấy dữ liệu kênh từ YouTube API."""
    try:
        request = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        )
        response = request.execute()
        
        if not response["items"]:
            return None
        
        item = response["items"][0]
        snippet = item["snippet"]
        stats = item["statistics"]
        
        return {
            "channel_id": channel_id,
            "channel_name": snippet.get("title", ""),
            "subscriber_count": int(stats.get("subscriberCount", 0)) if stats.get("subscriberCount") else None,
            "total_video_count": int(stats.get("videoCount", 0)),
            "channel_created_at": snippet.get("publishedAt", ""),
            "country_code": snippet.get("country", ""),
        }
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy dữ liệu kênh: {str(e)}")
        return None

def create_gauge_chart(score: float) -> go.Figure:
    """Tạo biểu đồ gauge cho điểm viral."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Điểm Viral (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 50], 'color': '#ffffcc'},
                {'range': [50, 75], 'color': '#ccffcc'},
                {'range': [75, 100], 'color': '#66ff66'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# ================= GIAO DIỆN =================

st.title("🤖 Dự Đoán Video Viral với Machine Learning")
st.caption("Sử dụng mô hình XGBoost đã được huấn luyện để dự đoán khả năng viral của video YouTube")

# Cảnh báo API quota
st.info("💡 **Lưu ý:** Chức năng này sử dụng YouTube API. Nếu gặp lỗi quota, vui lòng thử lại vào ngày mai.")

# Phần nhập liệu
st.subheader("📹 Nhập Video")

input_col1, input_col2 = st.columns([3, 1])

with input_col1:
    video_input = st.text_input(
        "Nhập URL hoặc Video ID của YouTube:",
        placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ hoặc dQw4w9WgXcQ",
        help="Hỗ trợ: youtube.com/watch?v=..., youtu.be/..., hoặc chỉ Video ID (11 ký tự)"
    )

with input_col2:
    st.write("")  # Khoảng đệm
    st.write("")  # Khoảng đệm
    predict_button = st.button("🔮 Dự Đoán", use_container_width=True, type="primary")

# Luồng dự đoán
if predict_button and video_input:
    video_id = extract_video_id(video_input)
    
    with st.spinner("⏳ Đang lấy dữ liệu từ YouTube..."):
        # Lấy dữ liệu video
        video_data = get_video_data(video_id)
        
        if not video_data:
            st.error("❌ **Không tìm thấy video!**")
            st.info("""
            **Có thể do:**
            - ✗ Video ID/URL không đúng
            - ✗ Video đã bị xóa hoặc ẩn
            - ✗ Video không public (private/unlisted)
            - ✗ API quota đã hết (xem lỗi chi tiết bên trên)
            
            **Hướng dẫn:**
            - ✓ Kiểm tra lại URL: `https://youtube.com/watch?v=...`
            - ✓ Hoặc chỉ dán Video ID (11 ký tự): `dQw4w9WgXcQ`
            - ✓ Đảm bảo video là public
            """)
            st.stop()
        
        # Lấy dữ liệu kênh
        channel_data = get_channel_data(video_data["channel_id"])
        
        if not channel_data:
            st.error("❌ Không thể lấy thông tin kênh.")
            st.stop()
    
    # Hiển thị thông tin video
    st.markdown("---")
    st.subheader("📺 Thông Tin Video")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("👁️ Lượt xem", f"{video_data['view_count']:,}")
    
    with col2:
        if video_data['like_count'] is not None:
            st.metric("👍 Lượt thích", f"{video_data['like_count']:,}")
        else:
            st.metric("👍 Lượt thích", "Ẩn")
    
    with col3:
        st.metric("💬 Bình luận", f"{video_data['comment_count']:,}")
    
    st.markdown(f"**Tiêu đề:** {video_data['title']}")
    st.markdown(f"**Kênh:** {channel_data['channel_name']} ({safe_int(channel_data.get('subscriber_count'), 0):,} subscribers)")
    st.markdown(f"**Ngày đăng:** {video_data['published_at']}")
    
    # Chạy dự đoán
    st.markdown("---")
    
    with st.spinner("🤖 Đang chạy mô hình dự đoán..."):
        try:
            result = predict_from_api_data(video_data, channel_data)
            
            # Hiển thị kết quả
            st.subheader("🎯 Kết Quả Dự Đoán")
            
            # Biểu đồ gauge và các chỉ số
            gauge_col, metrics_col = st.columns([1, 1])
            
            with gauge_col:
                fig = create_gauge_chart(result['viral_score'])
                st.plotly_chart(fig, use_container_width=True)
            
            with metrics_col:
                st.markdown("### 📊 Thống Kê")
                st.markdown(f"**Dự đoán:** `{result['prediction']}`")
                st.markdown(f"**Độ tin cậy:** `{result['confidence']}`")
                st.markdown(f"**Viral Score:** `{result['viral_score']:.2%}`")
                st.markdown(f"**Ngày đăng:** {result['days_since_published']} ngày trước")
            
            # Top 5 yếu tố ảnh hưởng
            st.markdown("---")
            st.subheader("⚡ Top 5 Yếu Tố Ảnh Hưởng")
            
            drivers_data = []
            for driver in result['top_5_drivers']:
                drivers_data.append({
                    "Feature": driver['feature'],
                    "Impact": driver['impact'],
                    "Ý Nghĩa": driver['meaning']
                })
            
            df_drivers = pd.DataFrame(drivers_data)
            
            # Tô màu tác động dương/âm
            def color_impact(val):
                if isinstance(val, (int, float)):
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}; font-weight: bold'
                return ''
            
            st.dataframe(
                df_drivers.style.applymap(color_impact, subset=['Impact']),
                use_container_width=True,
                hide_index=True
            )
            
            # Khuyến nghị
            st.markdown("---")
            st.subheader("💡 Đề Xuất")
            
            if result['viral_score'] >= 0.75:
                st.success(f"**{result['recommendation']}**")
            elif result['viral_score'] >= 0.50:
                st.info(f"**{result['recommendation']}**")
            elif result['viral_score'] >= 0.30:
                st.warning(f"**{result['recommendation']}**")
            else:
                st.error(f"**{result['recommendation']}**")
            
            # Cảnh báo
            if result.get('warnings'):
                st.markdown("---")
                st.subheader("⚠️ Cảnh Báo")
                for warning in result['warnings']:
                    st.warning(warning)
            
        except FileNotFoundError:
            st.error("❌ Model chưa được train. Vui lòng chạy `python -m ml.train` trước.")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {str(e)}")
            st.exception(e)

# Thông tin sidebar
with st.sidebar:
    st.markdown("### 📖 Hướng Dẫn")
    st.markdown("""
    **Cách sử dụng:**
    1. Nhập URL hoặc ID của video YouTube
    2. Nhấn nút "Dự Đoán"
    3. Xem kết quả phân tích
    
    **Điểm Viral:**
    - 🟢 75-100%: Rất có khả năng viral
    - 🟡 50-75%: Có tiềm năng viral
    - 🟠 30-50%: Tiềm năng thấp
    - 🔴 0-30%: Khó viral
    
    **Model:**
    - XGBoost Classifier
    - Huấn luyện trên dữ liệu YouTube
    - Sử dụng hơn 30 đặc trưng
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ Thông Tin Model")
    
    # Thử nạp thông tin model
    try:
        from ml.save_load import load_model
        model, config = load_model()
        st.success("✅ Model đã được load")
        if config.get("viral_threshold"):
            st.info(f"Ngưỡng: {config['viral_threshold']:.2f}")
    except:
        st.error("❌ Model chưa được train")

# Chân trang
st.markdown("---")
st.caption("YouTube Analytics Pipeline • ML Viral Prediction")
