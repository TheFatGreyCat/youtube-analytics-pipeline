import streamlit as st
from googleapiclient.discovery import build

# ================= CẤU HÌNH =================
from dotenv import load_dotenv
import os

# Nạp biến môi trường từ file .env ở thư mục gốc (nếu có)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, '.env'))

API_KEY = os.getenv('YOUTUBE_API_KEY') or ""
if not API_KEY:
    st.error("🔑 YOUTUBE_API_KEY chưa được thiết lập. Thiết lập trong .env hoặc biến môi trường.")
    st.stop()

# Khởi tạo client với kiểm tra an toàn
try:
    youtube = build("youtube", "v3", developerKey=API_KEY)
except Exception as err:
    st.error(
        "❌ Không thể khởi tạo YouTube client. Kiểm tra YOUTUBE_API_KEY hợp lệ."
        f" Chi tiết: {err}"
    )
    st.stop()

st.set_page_config(page_title="YouTube Search", layout="wide")

st.title("🔎 Tìm kiếm kênh YouTube")

query = st.text_input("Nhập từ khóa tìm kiếm")

def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

# ================= HÀM TÌM KIẾM =================
def search_youtube(keyword):
    try:
        request = youtube.search().list(
            q=keyword,
            part="snippet",
            type="channel",
            maxResults=5
        )
        response = request.execute()
        return response.get("items", [])
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            st.error("❌ Đã vượt quá quota YouTube API hôm nay. Vui lòng thử lại sau.")
        else:
            st.error(f"❌ Lỗi khi tìm kiếm kênh: {error_msg}")
        return []

# ================= NÚT TÌM KIẾM =================
if st.button("Tìm kiếm"):
    if query.strip() == "":
        st.warning("Vui lòng nhập từ khóa.")
    else:
        st.session_state.search_results = search_youtube(query)

# ================= HIỂN THỊ KẾT QUẢ =================
if "search_results" in st.session_state:

    for item in st.session_state.search_results:

        snippet_data = item.get("snippet", {})
        channel_id = snippet_data.get("channelId", "")
        channel_title = snippet_data.get("title", "Không rõ tên kênh")
        description = snippet_data.get("description", "")
        thumbnail = snippet_data.get("thumbnails", {}).get("high", {}).get("url", "")

        if not channel_id:
            continue

        # Lấy thống kê kênh
        stats = {}
        try:
            channel_request = youtube.channels().list(
                part="statistics",
                id=channel_id
            )
            channel_response = channel_request.execute()
            channel_items = channel_response.get("items", [])
            if channel_items:
                stats = channel_items[0].get("statistics", {})
        except Exception:
            pass

        subscribers = f"{safe_int(stats.get('subscriberCount'), 0):,}"
        total_views = f"{safe_int(stats.get('viewCount'), 0):,}"
        video_count = f"{safe_int(stats.get('videoCount'), 0):,}"

        st.markdown("---")

        col1, col2 = st.columns([1, 4])

        with col1:
            if thumbnail:
                st.image(thumbnail, width=120)

        with col2:
            st.subheader(channel_title)
            st.write(f"👥 Subscribers: {subscribers}")
            st.write(f"🎬 Videos: {video_count}")
            st.write(f"👁 Views: {total_views}")
            if description:
                st.write(description[:200] + "...")
            else:
                st.write("(Không có mô tả)")

            col_btn1, col_btn2 = st.columns(2)

            with col_btn1:
                st.markdown(
                    f"[👉 Xem kênh](https://www.youtube.com/channel/{channel_id})"
                )

            with col_btn2:
                if st.button("📊 Phân tích", key=f"analyze_{channel_id}"):
                    st.session_state.selected_channel_id = channel_id
                    st.session_state.selected_channel_name = channel_title
                    st.switch_page("app.py")