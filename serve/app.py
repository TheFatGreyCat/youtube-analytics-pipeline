import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import numpy as np
import os

# ================== CẤU HÌNH ==================
st.set_page_config(page_title="YouTube Analytics Dashboard", layout="wide")

# Cấu hình YouTube API
# Nạp file .env khi chạy local để sử dụng biến môi trường
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(project_root, '.env'))

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

# Đảm bảo có API key trước khi khởi tạo client
if not YOUTUBE_API_KEY:
    st.error("🔑 YouTube API key not found. Please set the YOUTUBE_API_KEY environment variable or add it to your .env file.")
    st.stop()

# Khởi tạo YouTube service object, xử lý fallback xác thực ngoài ý muốn
try:
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
except Exception as err:
    # Nếu thư viện vẫn thử dùng ADC và thất bại, hiển thị thông báo thân thiện
    st.error(
        "❌ Failed to initialise YouTube client. "
        "Ensure your API key is valid and you have network access. "
        f"Error details: {err}"
    )
    st.stop()

# ================== TIÊU ĐỀ ==================
st.title("📊 YouTube Analytics Dashboard")
st.caption("Phân tích chi tiết dữ liệu YouTube")

# ================== KÊNH ĐƯỢC CHỌN TỪ TÌM KIẾM ==================
channel_id = st.session_state.get("selected_channel_id")
channel_name = st.session_state.get("selected_channel_name", "Unknown")

if not channel_id:
    st.warning("🔎 Hãy tìm và chọn một kênh trước.")
    st.switch_page("pages/1_Channel_Search.py")
    st.stop()

st.success(f"🎯 Đang phân tích kênh: {channel_name}")
st.write(f"Channel ID: {channel_id}")

# Nút refresh dữ liệu real-time
col_refresh, col_space = st.columns([1, 10])
with col_refresh:
    if st.button("🔄 Refresh", help="Cập nhật dữ liệu từ YouTube"):
        st.rerun()

# ================== HÀM HỖ TRỢ ==================

def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

def calculate_engagement_rate(likes, comments, views):
    """Tính tỉ lệ engagement"""
    if views == 0:
        return 0
    return ((likes + comments) / views) * 100

def detect_trending_videos(videos_df, min_engagement_rate=2.0, recent_days=7):
    """Phát hiện video đang trending"""
    if videos_df.empty:
        return pd.DataFrame()
    
    df = videos_df.copy()
    df['published_at'] = pd.to_datetime(df['published_at'])
    
    # Tính engagement rate
    df['engagement_rate'] = df.apply(
        lambda row: calculate_engagement_rate(row['like_count'], row['comment_count'], row['view_count']),
        axis=1
    )
    
    # Lọc video có engagement rate cao
    trending_df = df[df['engagement_rate'] >= min_engagement_rate].copy()
    
    # Sắp xếp theo engagement rate
    trending_df = trending_df.sort_values('engagement_rate', ascending=False)
    
    return trending_df

def get_channel_stats(channel_id):
    """Lấy thống kê kênh từ YouTube API (Real-time, không cache)"""
    try:
        request = youtube.channels().list(
            part="statistics,snippet",
            id=channel_id
        )
        response = request.execute()
        
        if response["items"]:
            stats = response["items"][0]["statistics"]
            snippet = response["items"][0]["snippet"]
            return {
                "subscriber_count": safe_int(stats.get("subscriberCount", 0)),
                "total_views": safe_int(stats.get("viewCount", 0)),
                "total_videos": safe_int(stats.get("videoCount", 0)),
                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", "")
            }
    except Exception as e:
        st.error(f"❌ Lỗi API: {str(e)}")
        return None
    return None

def get_channel_videos(channel_id):
    """Lấy danh sách video từ YouTube API (Real-time, không cache)"""
    try:
        videos = []
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            order="viewCount",
            type="video",
            maxResults=50
        )
        
        response = request.execute()
        video_ids = [item["id"]["videoId"] for item in response.get("items", [])]
        
        if video_ids:
            stats_request = youtube.videos().list(
                part="statistics,snippet",
                id=",".join(video_ids[:20])
            )
            stats_response = stats_request.execute()
            
            for item in stats_response.get("items", []):
                video_id = item["id"]
                snippet = item["snippet"]
                stats = item["statistics"]
                
                videos.append({
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "view_count": safe_int(stats.get("viewCount", 0)),
                    "like_count": safe_int(stats.get("likeCount", 0)),
                    "comment_count": safe_int(stats.get("commentCount", 0))
                })
        
        df = pd.DataFrame(videos) if videos else pd.DataFrame()
        # Xác minh tính toán ở cấp video để đảm bảo độ chính xác
        if not df.empty:
            # Tính engagement rate và so sánh
            df['engagement_rate'] = df.apply(
                lambda r: calculate_engagement_rate(r['like_count'], r['comment_count'], r['view_count']),
                axis=1
            )
            # Công thức thủ công và ngưỡng sai số cho phép
            diffs = df.apply(
                lambda r: abs(r['engagement_rate'] - ((r['like_count'] + r['comment_count']) / (r['view_count'] if r['view_count']>0 else 1) * 100)),
                axis=1
            )
            if (diffs > 0.001).any():
                st.warning("⚠️ Phát hiện sai khác nhỏ trong tính toán engagement rate, vui lòng kiểm tra lại.")
        return df
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy video: {str(e)}")
        return pd.DataFrame()

# ================== NẠP DỮ LIỆU ==================
channel_stats = get_channel_stats(channel_id)
videos_df = get_channel_videos(channel_id)

if channel_stats is None:
    st.error("❌ Không thể lấy dữ liệu cho kênh này.")
    st.stop()

# ================== PHẦN 1: THẺ KPI ==================
st.subheader("📌 Chỉ số tổng quan")
col1, col2, col3, col4 = st.columns(4)

col1.metric("👥 Subscribers", f"{channel_stats['subscriber_count']:,}")
col2.metric("👁️ Total Views", f"{channel_stats['total_views']:,}")
col3.metric("🎬 Total Videos", f"{channel_stats['total_videos']:,}")

# Tính lượt xem trung bình mỗi video
avg_views = channel_stats['total_views'] // max(channel_stats['total_videos'], 1)
col4.metric("📊 Avg Views/Video", f"{avg_views:,}")

# ================== PHẦN 2: ẢNH ĐẠI DIỆN KÊNH ==================
if channel_stats["thumbnail"]:
    col_img, col_space = st.columns([1, 4])
    with col_img:
        st.image(channel_stats["thumbnail"], width=150)

# ================== PHẦN 3: PHÂN TÍCH VIDEO ==================
st.markdown("---")
st.subheader("📈 Phân tích Video")

if not videos_df.empty:
    # Thêm engagement rate
    videos_df['engagement_rate'] = videos_df.apply(
        lambda row: calculate_engagement_rate(row['like_count'], row['comment_count'], row['view_count']),
        axis=1
    )
    
    tab1, tab2, tab3 = st.tabs(["📊 Top Videos", "🔥 Trending Videos", "📉 Video Analytics"])
    
    with tab1:
        st.subheader("🎯 Top 20 Videos (Sắp xếp theo Views)")
        display_df = videos_df.copy()
        display_df['published_at'] = pd.to_datetime(display_df['published_at']).dt.strftime('%d/%m/%Y')
        display_df = display_df[['title', 'published_at', 'view_count', 'like_count', 'comment_count', 'engagement_rate']]
        display_df.columns = ['Title', 'Published', 'Views', 'Likes', 'Comments', 'Engagement %']
        display_df['Views'] = display_df['Views'].apply(lambda x: f"{x:,}")
        display_df['Likes'] = display_df['Likes'].apply(lambda x: f"{x:,}")
        display_df['Comments'] = display_df['Comments'].apply(lambda x: f"{x:,}")
        display_df['Engagement %'] = display_df['Engagement %'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(display_df, width='stretch', use_container_width=True)
    
    with tab2:
        st.subheader("🔥 Video Trending (Engagement Rate > 2%)")
        trending_df = detect_trending_videos(videos_df, min_engagement_rate=2.0)
        
        if not trending_df.empty:
            st.success(f"✅ Tìm thấy {len(trending_df)} video đang trending!")
            
            display_trending = trending_df.copy()
            display_trending['published_at'] = pd.to_datetime(display_trending['published_at']).dt.strftime('%d/%m/%Y')
            display_trending = display_trending[['title', 'published_at', 'view_count', 'engagement_rate']]
            display_trending.columns = ['Title', 'Published', 'Views', 'Engagement %']
            display_trending['Views'] = display_trending['Views'].apply(lambda x: f"{x:,}")
            display_trending['Engagement %'] = display_trending['Engagement %'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_trending, width='stretch', use_container_width=True)
            
            # Biểu đồ trending videos
            fig = px.bar(
                trending_df,
                x='engagement_rate',
                y='title',
                title='Trending Videos - Engagement Rate',
                labels={'engagement_rate': 'Engagement Rate (%)', 'title': ''},
                orientation='h'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Hiện tại không có video trending (Engagement Rate < 2%)")
    
    with tab3:
        st.subheader("📉 Biểu đồ Phân tích Video")
        
        # Biểu đồ Top 10 Views
        top_10 = videos_df.nlargest(10, 'view_count')
        fig1 = px.bar(
            top_10,
            x='view_count',
            y='title',
            title='Top 10 Videos - Views',
            labels={'view_count': 'Views', 'title': ''},
            orientation='h'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Biểu đồ Engagement Rate
        fig2 = px.scatter(
            videos_df,
            x='view_count',
            y='engagement_rate',
            size='like_count',
            hover_data=['title'],
            title='Views vs Engagement Rate',
            labels={'view_count': 'Views', 'engagement_rate': 'Engagement Rate (%)'}
        )
        st.plotly_chart(fig2, use_container_width=True)

# ================== PHẦN 4: CHỈ SỐ NÂNG CAO ==================
st.markdown("---")
st.subheader("📊 Chỉ số nâng cao")

# Cung cấp giải thích dễ hiểu cho từng chỉ số
st.markdown("""
**Giải thích các chỉ số:**

- **Engagement Rate trung bình:** trung bình tỷ lệ tương tác của các video, tính bằng `(likes + comments) / views * 100`.
- **Like Rate:** tỷ lệ lượt thích trên tổng lượt xem, thể hiện mức độ người xem bấm thích.
- **Comment Rate:** tỷ lệ bình luận trên tổng lượt xem, đo lường mức độ tương tác phản hồi.
- **Best Video:** video có tỷ lệ engagement cao nhất, thường là video hấp dẫn nhất.
""" )

if not videos_df.empty:
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    # Engagement rate trung bình
    avg_engagement = videos_df['engagement_rate'].mean() if 'engagement_rate' in videos_df.columns else 0
    col_m1.metric("Engagement trung bình", f"{avg_engagement:.2f}%")
    col_m1.write("(Likes+Comments)/Views * 100")
    
    # Tỷ lệ Like
    total_likes = videos_df['like_count'].sum()
    total_views = videos_df['view_count'].sum()
    like_rate = (total_likes / total_views * 100) if total_views > 0 else 0
    col_m2.metric("Tỷ lệ Like", f"{like_rate:.2f}%")
    col_m2.write("Likes / Views * 100")
    
    # Tỷ lệ Comment
    total_comments = videos_df['comment_count'].sum()
    comment_rate = (total_comments / total_views * 100) if total_views > 0 else 0
    col_m3.metric("Tỷ lệ Comment", f"{comment_rate:.2f}%")
    col_m3.write("Comments / Views * 100")
    
    # Video với engagement cao nhất
    if not videos_df.empty:
        best_video = videos_df.loc[videos_df['engagement_rate'].idxmax()]
        col_m4.metric("Video tốt nhất", str(best_video['title'])[:30] + "...")
        col_m4.write("Dựa trên engagement rate cao nhất")

# ================== CHÂN TRANG ==================
st.markdown("---")
st.caption("YouTube Analytics Pipeline • Streamlit • Powered by YouTube Data API")
st.markdown("---")
st.caption("YouTube Analytics Pipeline • Streamlit")