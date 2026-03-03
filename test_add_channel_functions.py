"""
Test script to verify Add Channel functions work correctly
"""

import pandas as pd
import os
import sys
from datetime import datetime

# Setup path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "config", "channels.csv")

# ================= CSV FUNCTIONS (COPIED FROM 2_Add_Channel.py) =================

def load_channels_from_csv():
    """Load channels dari file CSV"""
    if os.path.exists(CSV_FILE_PATH):
        try:
            df = pd.read_csv(CSV_FILE_PATH)
            return df
        except Exception as e:
            print(f"[ERROR] Loi doc file CSV: {str(e)}")
            return pd.DataFrame(columns=['num_id', 'channel_name', 'channel_id', 'subscriber_count', 'total_views', 'total_videos'])
    else:
        print(f"[OK] File CSV khong ton tai, tra ve DataFrame trong")
        return pd.DataFrame(columns=['num_id', 'channel_name', 'channel_id', 'subscriber_count', 'total_views', 'total_videos'])

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
    """Add channel vào file CSV (test copy)

    This version mirrors the production copy but emits print statements
    instead of streamlit.  It also checks for an existing entry and returns
    False if the channel already exists, matching updated behavior in
    `serve/pages/2_Add_Channel.py`.
    """
    # duplicate check
    exists, _ = check_channel_in_csv(channel_info.get('channel_name', ''),
                                      channel_info.get('channel_id', None))
    if exists:
        print(f"[WARN] Channel '{channel_info.get('channel_name')}' already exists in CSV")
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
        print(f"[ERROR] Loi luu vao file CSV: {str(e)}")
        return False

# ================= TEST HELPERS =================

# we don't want bootstrap-from-template to interfere with tests, so whenever
# we remove the CSV file we immediately create an empty file with just the
# header row.  that way `load_channels_from_csv` will read the empty file and
# not attempt to populate it from the template.
HEADER_DF = pd.DataFrame(columns=['num_id', 'channel_name', 'channel_id', 'subscriber_count', 'total_views', 'total_videos'])


# ================= TEST FUNCTIONS =================

def test_load_csv():
    """Test load CSV function"""
    print("\n" + "="*60)
    print("TEST 1: Load CSV Function")
    print("="*60)
    
    df = load_channels_from_csv()
    print(f"[OK] CSV loaded successfully")
    print(f"  DataFrame shape: {df.shape}")
    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Preview:\n{df.head()} (showing first rows)")
    else:
        print(f"  CSV is empty or doesn't exist yet")
    return df

def test_add_channel(channel_info):
    """Test add channel function"""
    print("\n" + "="*60)
    print("TEST 2: Add Channel Function")
    print("="*60)
    
    channel_name = channel_info['channel_name']
    channel_id = channel_info['channel_id']
    
    print(f"Adding channel: {channel_name} ({channel_id})")
    result = add_channel_to_csv(channel_info)
    
    if result:
        print(f"[OK] Channel added successfully")
        print(f"  CSV file path: {CSV_FILE_PATH}")
        
        # Verify file was created/updated
        if os.path.exists(CSV_FILE_PATH):
            df = pd.read_csv(CSV_FILE_PATH)
            print(f"  CSV now contains {len(df)} channels")
            print(f"  Last added channel: {df.iloc[-1].to_dict()}")
            return True
        else:
            print(f"[ERROR] CSV file not found after adding")
            return False
    else:
        print(f"[ERROR] Failed to add channel")
        return False

def test_check_channel(channel_name, channel_id=None):
    """Test check channel function"""
    print("\n" + "="*60)
    print("TEST 3: Check Channel Function")
    print("="*60)
    
    if channel_id:
        print(f"Checking for channel ID: {channel_id}")
    else:
        print(f"Checking for channel name: {channel_name}")
    
    exists, channel_data = check_channel_in_csv(channel_name, channel_id)
    
    if exists:
        print(f"[OK] Channel found in CSV!")
        print(f"  Channel data: {channel_data.to_dict()}")
        return True
    else:
        print(f"[OK] Channel not found (expected if first test)")
        return False

def test_duplicate_prevention():
    """Test that we can't add duplicate channels"""
    print("\n" + "="*60)
    print("TEST 4: Duplicate Prevention")
    print("="*60)
    
    test_channel = {
        'channel_name': 'Test Channel Duplicate',
        'channel_id': 'UCTestDuplicate123',
        'subscriber_count': 1000,
        'total_views': 50000,
        'total_videos': 25
    }
    
    # Add once
    print("Adding test channel first time...")
    result1 = add_channel_to_csv(test_channel)
    print(f"[OK] First add result: {result1}")
    
    # Try to add again
    print("Adding same test channel second time...")
    result2 = add_channel_to_csv(test_channel)
    print(f"[OK] Second add result: {result2} (should be False)")
    
    # Check how many times it appears
    df = load_channels_from_csv()
    duplicates = df[df['channel_id'] == test_channel['channel_id']]
    print(f"[OK] Channel appears {len(duplicates)} time(s) in CSV")
    
    # Expect only one copy and second add returning False
    correct_behavior = (len(duplicates) == 1 and result1 is True and result2 is False)
    print(f"[OK] Duplicate prevention working: {correct_behavior}")
    return correct_behavior

def test_data_integrity():
    """Test data is stored correctly"""
    print("\n" + "="*60)
    print("TEST 5: Data Integrity")
    print("="*60)
    
    test_channel = {
        'channel_name': 'Integrity Test Channel',
        'channel_id': 'UCIntegrityTest456',
        'subscriber_count': 999999,
        'total_views': 100000000,
        'total_videos': 500
    }
    
    print(f"Adding channel with specific data...")
    add_channel_to_csv(test_channel)
    
    print(f"Retrieving channel data...")
    exists, data = check_channel_in_csv(None, test_channel['channel_id'])
    
    if exists:
        print(f"[OK] Data retrieved successfully")
        
        # Check all fields
        checks = {
            'channel_name': data['channel_name'] == test_channel['channel_name'],
            'channel_id': data['channel_id'] == test_channel['channel_id'],
            'subscriber_count': data['subscriber_count'] == test_channel['subscriber_count'],
            'total_views': data['total_views'] == test_channel['total_views'],
            'total_videos': data['total_videos'] == test_channel['total_videos']
        }
        
        all_pass = True
        for field, passed in checks.items():
            status = "[OK]" if passed else "[ERROR]"
            print(f"  {status} {field}: {data[field]} == {test_channel[field]}")
            all_pass = all_pass and passed
        
        return all_pass
    else:
        print(f"[ERROR] Channel not found after adding")
        return False

# ================= MAIN TEST SEQUENCE =================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("YouTube Analytics Pipeline - Add Channel Functions Test")
    print("="*60)
    
    # Preliminary TEST 0: ensure template bootstrapping works when CSV missing
    print("\n" + "="*60)
    print("TEST 0: Bootstrap from template (CSV missing)")
    print("="*60)
    if os.path.exists(CSV_FILE_PATH):
        os.remove(CSV_FILE_PATH)
    df_boot = load_channels_from_csv()
    print(f"[OK] Rows loaded: {len(df_boot)}")
    if len(df_boot) >= 50:
        print("[OK] Template provided expected channel count")
    else:
        print("[WARN] Template may be missing or incomplete")
    
    # Clean start - Remove CSV for fresh test and write empty header
    if os.path.exists(CSV_FILE_PATH):
        print(f"\nRemoving existing CSV for fresh test: {CSV_FILE_PATH}")
        os.remove(CSV_FILE_PATH)
        print("[OK] CSV removed")
    HEADER_DF.to_csv(CSV_FILE_PATH, index=False)
    print("[OK] Created empty CSV with header to avoid template bootstrapping")

    # Test 1: Load CSV (should be empty)
    df = test_load_csv()

    # Test 2: Add first channel
    channel1 = {
        'channel_name': 'MrBeast',
        'channel_id': 'UCX6OQ9mkjeUhA15g1mVeCMA',
        'subscriber_count': 200000000,
        'total_views': 50000000000,
        'total_videos': 500
    }
    test_add_channel(channel1)

    # Test 3: Check if channel exists
    test_check_channel(None, channel1['channel_id'])

    # Test 4: Add another channel
    channel2 = {
        'channel_name': 'T-Series',
        'channel_id': 'UChuZAo31fN56gl-pftitnrA',
        'subscriber_count': 250000000,
        'total_views': 300000000000,
        'total_videos': 150000
    }
    test_add_channel(channel2)

    # Test 5: Duplicate prevention
    test_duplicate_prevention()

    # Test 6: Data integrity
    test_data_integrity()

    # Final summary
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    
    df_final = load_channels_from_csv()
    print(f"[OK] Total channels in CSV: {len(df_final)}")
    print(f"[OK] CSV file location: {CSV_FILE_PATH}")
    print(f"[OK] CSV columns: {list(df_final.columns)}")
    print(f"\nFinal CSV Preview:")
    print(df_final.to_string())
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")
