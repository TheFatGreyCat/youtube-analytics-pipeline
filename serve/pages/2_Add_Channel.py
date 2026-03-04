import streamlit as st
from googleapiclient.discovery import build
import pandas as pd
import os
from dotenv import load_dotenv

# ================= CẤU HÌNH =================
# Nạp biến môi trường từ thư mục gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# Đường dẫn file CSV
CSV_PATH = os.path.join(BASE_DIR, 'config', 'channels.csv')
TEMPLATE_CSV_PATH = os.path.join(BASE_DIR, 'config', 'channels_template.csv')

API_KEY = os.getenv('YOUTUBE_API_KEY') or ""
if not API_KEY:
    st.error("🔑 YOUTUBE_API_KEY chưa được thiết lập. Thêm vào .env hoặc biến môi trường và khởi động lại.")
    st.stop()

# Khởi tạo YouTube client (tránh vô tình dùng ADC)
try:
    youtube = build("youtube", "v3", developerKey=API_KEY)
except Exception as err:
    st.error(
        "❌ Không thể khởi tạo YouTube client. Xác nhận YOUTUBE_API_KEY hợp lệ."
        f" Chi tiết: {err}"
    )
    st.stop()

st.set_page_config(page_title="Add Channel", layout="wide")

st.title("➕ Thêm kênh mới")
st.caption("Thêm kênh mới vào hệ thống để phân tích")

# ================= CÁC HÀM =================

def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

def load_channels_csv():
    """Load danh sách channels từ CSV"""
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception:
            df = pd.DataFrame()
        required_cols = ['num_id', 'channel_name', 'channel_id', 'subscriber_count', 'total_views', 'total_videos']
        for col in required_cols:
            if col not in df.columns:
                df[col] = '' if col in ['channel_name', 'channel_id'] else 0
        return df[required_cols]
    elif os.path.exists(TEMPLATE_CSV_PATH):
        # Khởi tạo từ template nếu channels.csv chưa tồn tại
        df = pd.read_csv(TEMPLATE_CSV_PATH)
        df.to_csv(CSV_PATH, index=False)
        return df
    else:
        # Tạo DataFrame rỗng với đúng các cột
        return pd.DataFrame(columns=['num_id', 'channel_name', 'channel_id', 'subscriber_count', 'total_views', 'total_videos'])

def save_channels_csv(df):
    """Lưu danh sách channels vào CSV"""
    df.to_csv(CSV_PATH, index=False)
    st.success(f"✅ Đã lưu vào {CSV_PATH}")

def save_channels_template_entry(num_id, channel_name):
    """Ghi num_id và channel_name vào channels_template.csv"""
    template_cols = ['num_id', 'channel_name']
    if os.path.exists(TEMPLATE_CSV_PATH):
        try:
            template_df = pd.read_csv(TEMPLATE_CSV_PATH)
        except Exception:
            template_df = pd.DataFrame(columns=template_cols)
    else:
        template_df = pd.DataFrame(columns=template_cols)

    for col in template_cols:
        if col not in template_df.columns:
            template_df[col] = ''

    template_df = template_df[template_cols]

    match_idx = template_df[template_df['num_id'].astype(str) == str(num_id)].index
    if len(match_idx) > 0:
        template_df.at[match_idx[0], 'channel_name'] = channel_name
    else:
        new_template_row = pd.DataFrame([{
            'num_id': num_id,
            'channel_name': channel_name,
        }])
        template_df = pd.concat([template_df, new_template_row], ignore_index=True)

    template_df['num_id'] = pd.to_numeric(template_df['num_id'], errors='coerce').fillna(0).astype(int)
    template_df = template_df.sort_values(by='num_id', ascending=True, kind='stable')
    template_df.to_csv(TEMPLATE_CSV_PATH, index=False)

def get_channel_by_channel_id(channel_id):
    """Lấy thông tin kênh bằng Channel ID"""
    try:
        request = youtube.channels().list(
            part="statistics,snippet",
            id=channel_id
        )
        response = request.execute()
        
        if response["items"]:
            item = response["items"][0]
            stats = item["statistics"]
            snippet = item["snippet"]
            
            return {
                "channel_id": channel_id,
                "channel_name": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "subscriber_count": int(stats.get("subscriberCount", 0)) if stats.get("subscriberCount") else 0,
                "total_views": int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                "total_videos": int(stats.get("videoCount", 0)) if stats.get("videoCount") else 0
            }
    except Exception as e:
        st.error(f"❌ Lỗi: {str(e)}")
        return None

def search_channels_by_name(keyword):
    """Tìm kiếm kênh bằng tên"""
    try:
        request = youtube.search().list(
            part="snippet",
            q=keyword,
            type="channel",
            maxResults=10
        )
        response = request.execute()
        channels = []
        
        for item in response.get("items", []):
            channel_id = item["snippet"]["channelId"]
            channel_info = get_channel_by_channel_id(channel_id)
            if channel_info:
                channels.append(channel_info)
        
        return channels
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            st.error("❌ Đã vượt quá giới hạn API hôm nay. Vui lòng thử lại vào ngày mai hoặc cập nhật quota.")
        else:
            st.error(f"❌ Lỗi tìm kiếm: {error_msg}")
        return []

def add_or_update_channel_csv(channel_info):
    """Thêm hoặc cập nhật kênh trong CSV"""
    try:
        # Nạp danh sách kênh hiện tại
        channels_df = load_channels_csv()
        
        channel_id = channel_info["channel_id"]
        
        # Kiểm tra kênh đã tồn tại hay chưa
        existing = channels_df[channels_df['channel_id'] == channel_id]
        
        if not existing.empty:
            # Cập nhật kênh đã có
            idx = existing.index[0]
            channels_df.at[idx, 'channel_name'] = channel_info["channel_name"]
            channels_df.at[idx, 'subscriber_count'] = channel_info["subscriber_count"]
            channels_df.at[idx, 'total_views'] = channel_info["total_views"]
            channels_df.at[idx, 'total_videos'] = channel_info["total_videos"]
            
            save_channels_csv(channels_df)
            return True
        else:
            # Thêm kênh mới
            current_max_id = pd.to_numeric(channels_df.get('num_id', pd.Series(dtype='int64')), errors='coerce').max()
            new_id = int(current_max_id) + 1 if pd.notna(current_max_id) else 1
            new_row = pd.DataFrame([{
                'num_id': new_id,
                'channel_name': channel_info["channel_name"],
                'channel_id': channel_id,
                'subscriber_count': channel_info["subscriber_count"],
                'total_views': channel_info["total_views"],
                'total_videos': channel_info["total_videos"]
            }])
            
            channels_df = pd.concat([channels_df, new_row], ignore_index=True)
            save_channels_csv(channels_df)
            save_channels_template_entry(new_id, channel_info["channel_name"])
            return True
            
    except Exception as e:
        st.error(f"❌ Lỗi lưu dữ liệu: {str(e)}")
        return False

# ================= KHỞI TẠO SESSION STATE =================
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'last_search_keyword' not in st.session_state:
    st.session_state.last_search_keyword = ""
if 'channel_added' not in st.session_state:
    st.session_state.channel_added = False

# ================= PHẦN GIAO DIỆN 1: TÌM THEO TÊN =================

st.subheader("🔍 Nhập tên kênh bạn muốn thêm vào danh sách")

# Form tìm kiếm
with st.form("search_form"):
    search_keyword = st.text_input("Nhập tên kênh", value=st.session_state.last_search_keyword)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_submitted = st.form_submit_button("🔎 Tìm kiếm", use_container_width=True)

if search_submitted and search_keyword.strip():
    st.session_state.last_search_keyword = search_keyword
    with st.spinner("Đang tìm kiếm trên YouTube..."):
        channels = search_channels_by_name(search_keyword)
        st.session_state.search_results = channels
        
        if channels:
            st.success(f"✅ Tìm thấy {len(channels)} kênh!")
        else:
            st.warning("❌ Không tìm thấy kênh nào! Hãy thử từ khóa khác.")

# Hiển thị kết quả tìm kiếm đã lưu
if st.session_state.search_results:
    st.subheader("📺 Kết quả tìm kiếm")
    
    for i, channel in enumerate(st.session_state.search_results, 1):
        col_left, col_right = st.columns([2, 3])
        
        with col_left:
            if channel['thumbnail']:
                st.image(channel['thumbnail'], width=200)
        
        with col_right:
            st.markdown(f"### {i}. {channel['channel_name']}")
            st.markdown(f"**Channel ID:** `{channel['channel_id']}`")
            st.markdown(f"**Subscribers:** {channel['subscriber_count']:,}")
            st.markdown(f"**Total Views:** {channel['total_views']:,}")
            st.markdown(f"**Total Videos:** {channel['total_videos']:,}")
            description_text = channel.get('description') or ''
            st.markdown(f"**Description:** {(description_text[:150] + '...') if description_text else '(Không có mô tả)'}")
            
            # Kiểm tra kênh đã tồn tại hay chưa
            channels_df = load_channels_csv()
            existing_row = channels_df[channels_df['channel_id'] == channel['channel_id']]
            exists = not existing_row.empty
            
            # Hiển thị thông báo nếu kênh đã có trong danh sách
            if exists:
                existing_info = existing_row.iloc[0]
                st.warning(
                    f"⚠️ **Kênh này đã có trong danh sách!**\n\n"
                    f"- Được thêm lúc: #ID {int(existing_info['num_id'])}\n"
                    f"- Subscribers hiện tại: {safe_int(existing_info['subscriber_count']):,}\n"
                    f"- Views: {safe_int(existing_info['total_views']):,}\n\n"
                    f"Nhấn nút dưới để **cập nhật** thông tin mới nhất từ YouTube."
                )
            else:
                st.info(
                    "✨ **Đây là một kênh mới!**\n\n"
                    "Kênh này chưa có trong danh sách. Nhấn nút dưới để thêm vào."
                )
            
            button_label = "♻️ Cập nhật thông tin" if exists else "✅ Thêm kênh này"
            
            if st.button(button_label, key=f"add_btn_{i}", use_container_width=True):
                if add_or_update_channel_csv(channel):
                    # Đánh dấu đã thêm/cập nhật để làm mới danh sách
                    st.session_state.channel_added = True
                    # Xóa kết quả tìm kiếm ngay lập tức
                    st.session_state.search_results = []
                    st.session_state.last_search_keyword = ""
                    # Chạy lại để cập nhật danh sách
                    st.rerun()
        
        st.divider()


# ================= PHẦN GIAO DIỆN 3: QUẢN LÝ KÊNH =================
st.markdown("---")

# Header, bộ lọc sắp xếp và nút nằm cùng một hàng
col_header, col_sort, col_button = st.columns([2.2, 1.3, 1])

with col_header:
    st.subheader("📋 Danh sách kênh đã thêm")

with col_sort:
    st.write("")
    newest_first_label = "Mới thêm gần đây (ID giảm dần)"
    if st.session_state.channel_added:
        st.session_state.channel_sort_option = newest_first_label
    sort_option = st.selectbox(
        "Sắp xếp",
        [
            newest_first_label,
            "Subscriber giảm dần",
            "Views giảm dần",
        ],
        key="channel_sort_option",
    )

with col_button:
    st.write("")  # Khoảng đệm để căn hàng
    st.write("")  # Khoảng đệm để căn hàng
    refresh_clicked = st.button("🔄 Cập nhật thống kê", use_container_width=True)

# Hiển thị thông báo thành công nếu vừa thêm/cập nhật kênh
if st.session_state.channel_added:
    st.success("🎉 Kênh đã được thêm/cập nhật thành công! Danh sách dưới đây đã cập nhật.")
    st.session_state.channel_added = False

try:
    channels_df = load_channels_csv()
    
    if not channels_df.empty:
        # Lọc bỏ các dòng có channel_id trống
        channels_df_filtered = channels_df[channels_df['channel_id'].notna() & (channels_df['channel_id'] != '')]
        
        if not channels_df_filtered.empty:
            sorted_df = channels_df_filtered.copy()
            for col in ['num_id', 'subscriber_count', 'total_views', 'total_videos']:
                sorted_df[col] = pd.to_numeric(sorted_df[col], errors='coerce').fillna(0)

            sort_config = {
                "Mới thêm gần đây (ID giảm dần)": ("num_id", False),
                "Subscriber giảm dần": ("subscriber_count", False),
                "Views giảm dần": ("total_views", False),
            }

            sort_col, ascending = sort_config[sort_option]
            sorted_df = sorted_df.sort_values(by=sort_col, ascending=ascending, kind='stable')

            display_df = sorted_df.copy()
            # Định dạng số để hiển thị
            display_df['subscriber_count'] = display_df['subscriber_count'].apply(lambda x: f"{safe_int(x):,}")
            display_df['total_views'] = display_df['total_views'].apply(lambda x: f"{safe_int(x):,}")
            
            st.dataframe(display_df, use_container_width=True)
            st.caption(f"📊 Tổng cộng: {len(channels_df_filtered)} kênh")
            
            # Xử lý khi bấm nút cập nhật thống kê
            if refresh_clicked:
                st.info("⏳ Đang cập nhật thông tin từ YouTube...")
                updated_count = 0
                
                for idx, row in channels_df_filtered.iterrows():
                    channel_id = row['channel_id']
                    channel_info = get_channel_by_channel_id(channel_id)
                    
                    if channel_info:
                        # Cập nhật dòng dữ liệu tương ứng
                        channels_df.at[idx, 'subscriber_count'] = channel_info['subscriber_count']
                        channels_df.at[idx, 'total_views'] = channel_info['total_views']
                        channels_df.at[idx, 'total_videos'] = channel_info['total_videos']
                        updated_count += 1
                
                save_channels_csv(channels_df)
                st.success(f"✅ Đã cập nhật {updated_count} kênh")
                st.rerun()
        else:
            st.info("📭 Chưa có kênh nào được thêm vào hệ thống.")
    else:
        st.info("📭 Chưa có kênh nào được thêm vào hệ thống.")
        
except Exception as e:
    st.error(f"❌ Lỗi tải danh sách kênh: {str(e)}")

# ================= CHÂN TRANG =================
st.markdown("---")
st.caption("YouTube Analytics Pipeline • Channel Management")
