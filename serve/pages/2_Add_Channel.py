import streamlit as st
from googleapiclient.discovery import build
from google.cloud import bigquery
import pandas as pd
import os
import csv

# ================= CONFIG =================
API_KEY = "AIzaSyC0gDJ5ipTodDrMGHF2-Zg0qMftp_2UY6E"
youtube = build("youtube", "v3", developerKey=API_KEY)

st.set_page_config(page_title="Add Channel", layout="wide")

st.title("➕ Thêm kênh mới")
st.caption("Thêm kênh mới vào hệ thống để phân tích")

# Check BigQuery availability
st.markdown("---")
with st.expander("ℹ️ Hướng dẫn sử dụng", expanded=False):
    st.info("""
    **Cách sử dụng:**
    1. **Kiểm tra kênh** - Section 0: Kiểm tra xem kênh đã có trong danh sách chưa
    2. **Tìm kiếm** - Section 1: Tìm kiếm kênh theo tên rồi thêm
    3. **Thêm bằng ID** - Section 2: Nhập Channel ID (bắt đầu UC) để thêm trực tiếp
    4. **Xem danh sách** - Section 3: Xem tất cả kênh đã thêm
    
    **Lưu ý:** Kênh được lưu vào file CSV (config/channels.csv) để lưu trữ offline.
    """)

st.markdown("---")

# ================= BIGQUERY CLIENT =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
credentials_path = os.path.join(
    BASE_DIR,
    "credentials",
    "project-8fd99edc-9e20-4b82-b43-41fc5f2ccbcd.json"
)

# Check BigQuery availability
BIGQUERY_AVAILABLE = False
client = None
try:
    client = bigquery.Client.from_service_account_json(
        credentials_path,
        project="project-8fd99edc-9e20-4b82-b43"
    )
    # Test connection
    client.list_tables("raw")
    BIGQUERY_AVAILABLE = True
except Exception as bq_init_error:
    st.warning(f"⚠️ BigQuery không khả dụng: {str(bq_init_error)[:80]}")
    st.info("💡 App sẽ sử dụng CSV để lưu danh sách kênh")

# ================= CSV FILE PATH =================
CSV_FILE_PATH = os.path.join(BASE_DIR, "config", "channels.csv")

# ================= CSV FUNCTIONS =================

def load_channels_from_csv():
    """Load channels dari file CSV.

    If the CSV does not exist (or is empty) we attempt to bootstrap it from
    the `channels_template.csv` file located in the same config directory.
    The template contains only `num_id` and `channel_name`; remaining
    columns are filled with defaults so downstream code can rely on a
    consistent schema.  The initial copy is persisted to `channels.csv`
    so users can modify or extend it later.
    """
    def _make_empty():
        return pd.DataFrame(
            columns=['num_id', 'channel_name', 'channel_id',
                     'subscriber_count', 'total_views', 'total_videos']
        )

    if os.path.exists(CSV_FILE_PATH):
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            return df
        except Exception as e:
            st.error(f"❌ Lỗi đọc file CSV: {str(e)}")
            return _make_empty()

    # CSV does not exist yet; bootstrap from template if available
    template_path = os.path.join(BASE_DIR, "config", "channels_template.csv")
    if os.path.exists(template_path):
        try:
            templ = pd.read_csv(template_path)
            # ensure required columns
            for col in ['channel_id', 'subscriber_count', 'total_views', 'total_videos']:
                if col not in templ.columns:
                    templ[col] = 0 if col != 'channel_id' else ''
            templ = templ[['num_id', 'channel_name', 'channel_id',
                           'subscriber_count', 'total_views', 'total_videos']]

            os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
            templ.to_csv(CSV_FILE_PATH, index=False)
            return templ
        except Exception as e:
            st.error(f"❌ Lỗi đọc file template: {str(e)}")
            return _make_empty()

    # no CSV and no template
    return _make_empty()

def check_channel_in_csv(channel_name, channel_id=None):
    """Check xem channel đã có trong file CSV chưa"""
    df = load_channels_from_csv()
    if df.empty:
        return False, None
    
    if channel_id:
        result = df[df['channel_id'].astype(str).str.strip() == channel_id.strip()]
    else:
        result = df[df['channel_name'].str.lower() == channel_name.lower()]
    
    if not result.empty:
        return True, result.iloc[0]
    return False, None

def add_channel_to_csv(channel_info):
    """Add channel vào file CSV

    Before appending, check whether the channel already exists in the CSV.
    If it does, warn and return False so callers know nothing was added.
    This makes the CSV the canonical source-of-truth for channel list.
    """
    # first check duplicates by id or name
    exists, existing = check_channel_in_csv(
        channel_info.get('channel_name', ''),
        channel_info.get('channel_id', None)
    )
    if exists:
        st.warning(f"⚠️ Kênh '{channel_info.get('channel_name')}' đã tồn tại trong CSV.")
        return False

    try:
        df = load_channels_from_csv()
        
        # Get next num_id
        if df.empty:
            next_id = 1
        else:
            next_id = int(df['num_id'].max()) + 1
        
        # Create new row
        new_row = {
            'num_id': next_id,
            'channel_name': channel_info['channel_name'],
            'channel_id': channel_info['channel_id'],
            'subscriber_count': channel_info['subscriber_count'],
            'total_views': channel_info['total_views'],
            'total_videos': channel_info['total_videos']
        }
        
        # Append to dataframe
        new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
        
        # Save to CSV
        new_df.to_csv(CSV_FILE_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"❌ Lỗi lưu vào file CSV: {str(e)}")
        return False

# ================= FUNCTIONS =================

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
                "subscriber_count": int(stats.get("subscriberCount", 0)),
                "total_views": int(stats.get("viewCount", 0)),
                "total_videos": int(stats.get("videoCount", 0))
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
        st.error(f"❌ Lỗi tìm kiếm: {str(e)}")
        return []

def save_channel_to_bigquery(channel_info):
    """Lưu thông tin kênh vào BigQuery và CSV

    The CSV is always authoritative.  Before doing any inserts we verify the
    channel isn't already present in the CSV file; if it is we warn and bail
    out.  BigQuery is then checked separately but only when it's available.
    """
    # CSV duplicate check
    exists, _ = check_channel_in_csv(channel_info.get('channel_name', ''),
                                      channel_info.get('channel_id', None))
    if exists:
        st.warning(f"⚠️ Kênh '{channel_info.get('channel_name')}' đã tồn tại trong CSV. Không thêm nữa.")
        return False

    try:
        # Try to check if channel exists in BigQuery (if available)
        if BIGQUERY_AVAILABLE:
            try:
                query = f"""
                SELECT COUNT(*) as count FROM `project-8fd99edc-9e20-4b82-b43.raw.raw_channels`
                WHERE channel_id = '{channel_info["channel_id"]}'
                """
                result = client.query(query).to_dataframe()
                
                if result['count'].values[0] > 0:
                    st.warning(f"⚠️ Kênh '{channel_info['channel_name']}' đã tồn tại trong hệ thống!")
                    return False
            except Exception as bq_check_error:
                st.warning(f"⚠️ Không thể kiểm tra BigQuery: {str(bq_check_error)[:60]}")
        
        # Try to insert to BigQuery if available
        if BIGQUERY_AVAILABLE:
            try:
                from datetime import datetime
                insert_query = f"""
                INSERT INTO `project-8fd99edc-9e20-4b82-b43.raw.raw_channels`
                (channel_id, channel_name, description, subscriber_count, total_views, total_videos, thumbnail_url, extracted_at)
                VALUES (
                    '{channel_info["channel_id"]}',
                    '{channel_info["channel_name"].replace("'", "")}',
                    '{channel_info["description"].replace("'", "")}',
                    {channel_info["subscriber_count"]},
                    {channel_info["total_views"]},
                    {channel_info["total_videos"]},
                    '{channel_info["thumbnail"]}',
                    CURRENT_TIMESTAMP()
                )
                """
                
                client.query(insert_query).result()
                st.success("✅ Đã lưu vào BigQuery")
            except Exception as bq_insert_error:
                st.warning(f"⚠️ Lưu vào BigQuery thất bại: {str(bq_insert_error)[:60]}")
        else:
            st.info("ℹ️ Sẽ lưu vào CSV (BigQuery không khả dụng)")
        
        # Save to CSV (always do this)
        add_channel_to_csv(channel_info)
        st.success("✅ Đã lưu vào CSV")
        
        return True
    except Exception as e:
        st.error(f"❌ Lỗi lưu dữ liệu: {str(e)}")
        return False

# ================= UI SECTION 0: CHECK CHANNEL IN CSV =================
st.subheader("📂 Kiểm tra kênh trong file danh sách")
col_check = st.columns(1)[0]
check_channel_id = col_check.text_input("Nhập Channel ID hoặc tên kênh để kiểm tra")

if st.button("🔍 Kiểm tra"):
    if check_channel_id.strip():
        exists, channel_data = check_channel_in_csv(check_channel_id, check_channel_id)
        
        if exists:
            st.success("✅ Kênh đã có trong danh sách!")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Thông tin có trong file:**")
            with col2:
                st.write(f"**Tên:** {channel_data['channel_name']}")
                st.write(f"**Channel ID:** {channel_data['channel_id']}")
                st.write(f"**Subscribers:** {channel_data['subscriber_count']:,}")
                st.write(f"**Total Views:** {channel_data['total_views']:,}")
                st.write(f"**Total Videos:** {channel_data['total_videos']:,}")
        else:
            st.info("📝 Kênh chưa có trong danh sách. Bạn có thể thêm nó bằng các phương thức dưới đây.")
    else:
        st.warning("Hãy nhập Channel ID hoặc tên kênh!")

# ================= UI SECTION 1: SEARCH BY NAME =================
st.subheader("🔍 Tìm kiếm theo tên kênh")
col_search = st.columns(1)[0]
search_keyword = col_search.text_input("Nhập tên kênh hoặc từ khóa tìm kiếm")

if st.button("🔎 Tìm kiếm"):
    if search_keyword.strip():
        with st.spinner("Đang tìm kiếm..."):
            channels = search_channels_by_name(search_keyword)
            
            if channels:
                st.success(f"✅ Tìm thấy {len(channels)} kênh!")
                
                for i, channel in enumerate(channels, 1):
                    with st.expander(f"{i}. {channel['channel_name']}"):
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            if channel['thumbnail']:
                                st.image(channel['thumbnail'], width=150)
                        
                        with col2:
                            st.write(f"**Channel ID:** {channel['channel_id']}")
                            st.write(f"**Subscribers:** {channel['subscriber_count']:,}")
                            st.write(f"**Total Views:** {channel['total_views']:,}")
                            st.write(f"**Total Videos:** {channel['total_videos']:,}")
                            st.write(f"**Description:** {channel['description'][:200]}...")
                            
                            if st.button(f"✅ Thêm kênh này", key=f"add_{i}"):
                                if save_channel_to_bigquery(channel):
                                    st.success(f"✅ Đã thêm kênh '{channel['channel_name']}' vào hệ thống!")
                                    st.session_state.selected_channel_id = channel['channel_id']
                                    st.session_state.selected_channel_name = channel['channel_name']
                                    st.rerun()
                                else:
                                    st.info("ℹ️ Kênh đã tồn tại hoặc không thể lưu vào hệ thống.")
            else:
                st.warning("❌ Không tìm thấy kênh nào!")
    else:
        st.warning("Hãy nhập từ khóa tìm kiếm!")

# ================= UI SECTION 2: ADD BY CHANNEL ID =================
st.markdown("---")
st.subheader("🔗 Thêm bằng Channel ID")

channel_id_input = st.text_input("Nhập Channel ID (bắt đầu bằng UC...)")

if st.button("➕ Thêm kênh"):
    if channel_id_input.strip():
        with st.spinner("Đang tải thông tin kênh..."):
            channel_info = get_channel_by_channel_id(channel_id_input)
            
            if channel_info:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.image(channel_info['thumbnail'], width=150)
                
                with col2:
                    st.write(f"**Tên kênh:** {channel_info['channel_name']}")
                    st.write(f"**Subscribers:** {channel_info['subscriber_count']:,}")
                    st.write(f"**Total Views:** {channel_info['total_views']:,}")
                    st.write(f"**Total Videos:** {channel_info['total_videos']:,}")
                
                if st.button("✅ Xác nhận thêm kênh"):
                    if save_channel_to_bigquery(channel_info):
                        st.success(f"✅ Đã thêm kênh '{channel_info['channel_name']}' vào hệ thống!")
                        st.session_state.selected_channel_id = channel_info['channel_id']
                        st.session_state.selected_channel_name = channel_info['channel_name']
                        st.rerun()
                    else:
                        st.info("ℹ️ Kênh đã tồn tại hoặc không thể lưu vào hệ thống.")
            else:
                st.error("❌ Không tìm thấy kênh với ID này!")
    else:
        st.warning("Hãy nhập Channel ID!")

# ================= UI SECTION 3: MANAGE CHANNELS =================
st.markdown("---")
st.subheader("📋 Danh sách kênh đã thêm")

# Show channels from CSV
with st.expander("📂 Danh sách từ file config (channels.csv)", expanded=False):
    csv_df = load_channels_from_csv()
    if not csv_df.empty:
        # Format display dataframe
        display_csv_df = csv_df.copy()
        if 'subscriber_count' in display_csv_df.columns:
            display_csv_df['subscriber_count'] = display_csv_df['subscriber_count'].apply(lambda x: f"{x:,}")
            display_csv_df['total_views'] = display_csv_df['total_views'].apply(lambda x: f"{x:,}")
        st.dataframe(display_csv_df, width='stretch')
        st.caption(f"📊 Tổng cộng từ CSV: {len(csv_df)} kênh")
    else:
        st.info("📭 File danh sách chưa có kênh nào.")

# Show channels from BigQuery
st.write("**Danh sách từ BigQuery:**")

if BIGQUERY_AVAILABLE:
    try:
        query = """
        SELECT channel_id, channel_name, subscriber_count, total_views, total_videos, extracted_at
        FROM `project-8fd99edc-9e20-4b82-b43.raw.raw_channels`
        ORDER BY extracted_at DESC
        LIMIT 50
        """
        channels_df = client.query(query).to_dataframe()
        
        if not channels_df.empty:
            display_df = channels_df.copy()
            display_df['subscriber_count'] = display_df['subscriber_count'].apply(lambda x: f"{x:,}")
            display_df['total_views'] = display_df['total_views'].apply(lambda x: f"{x:,}")
            display_df['extracted_at'] = pd.to_datetime(display_df['extracted_at']).dt.strftime('%d/%m/%Y %H:%M')
            
            st.dataframe(display_df, width='stretch')
            st.caption(f"📊 Tổng cộng: {len(channels_df)} kênh")
        else:
            st.info("📭 Chưa có kênh nào được thêm vào hệ thống.")
    except Exception as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            st.warning(f"⚠️ BigQuery dataset chưa được tạo.")
        else:
            st.warning(f"⚠️ Không thể tải danh sách: {error_msg[:80]}")
else:
    st.info("ℹ️ BigQuery không khả dụng. Sử dụng CSV để lưu thay thế.")

# ================= FOOTER =================
st.markdown("---")
st.caption("YouTube Analytics Pipeline • Channel Management")
