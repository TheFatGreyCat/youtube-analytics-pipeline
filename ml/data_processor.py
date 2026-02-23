"""
Data Processing for Viral Prediction Model
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.fetch_intermediate import IntermediateLayerFetcher
from ml import config


class DataProcessor:
    """Process and prepare data for training"""
    
    def __init__(self, viral_percentile: float = None):
        """
        Args:
            viral_percentile: Percentile threshold for viral (default from config)
        """
        self.viral_percentile = viral_percentile or config.VIRAL_PERCENTILE
        self.feature_cols = None
        self.viral_threshold = None
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch training data from BigQuery"""
        print("\n" + "=" * 70)
        print("📥 FETCHING DATA FROM ALL INTERMEDIATE TABLES")
        print("=" * 70)
        fetcher = IntermediateLayerFetcher()
        df = fetcher.get_all_intermediate_data(limit=config.MAX_ROWS)
        print(f"\n✅ Successfully fetched {len(df):,} videos with {len(df.columns)} features")
        print(f"   📊 Data sources: int_engagement_metrics + int_videos__enhanced + int_channel_summary")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target variable
        
        Returns:
            tuple: (X, y) features and target
        """
        print("\n" + "=" * 70)
        print("⚙️  PREPARING FEATURES")
        print("=" * 70)
        
        # Define viral threshold
        self.viral_threshold = df['engagement_score'].quantile(self.viral_percentile)
        df['is_viral'] = (df['engagement_score'] >= self.viral_threshold).astype(int)
        
        viral_count = df['is_viral'].sum()
        not_viral_count = len(df) - viral_count
        
        # Data distribution table
        dist_table = [
            ["Not Viral", not_viral_count, f"{not_viral_count/len(df)*100:.1f}%"],
            ["Viral", viral_count, f"{viral_count/len(df)*100:.1f}%"],
            ["Total", len(df), "100.0%"],
            ["Viral Threshold", f"{self.viral_threshold:.2f}", f"Top {(1-self.viral_percentile)*100:.0f}%"]
        ]
        
        print("\n📊 Data Distribution:")
        print(tabulate(dist_table, headers=["Category", "Count", "Percentage"], tablefmt="pretty"))
        
        # Filter available columns from config
        self.feature_cols = [col for col in config.FEATURE_COLUMNS if col in df.columns]
        
        # Drop rows with missing values
        df_clean = df[self.feature_cols + ['is_viral']].dropna()
        
        # Features table
        features_table = [[i+1, feat] for i, feat in enumerate(self.feature_cols)]
        print("\n🎯 Selected Features:")
        print(tabulate(features_table, headers=["#", "Feature Name"], tablefmt="pretty"))
        print(f"\n✅ Clean data: {len(df_clean):,} rows (removed {len(df)-len(df_clean)} rows with missing values)")
        
        X = df_clean[self.feature_cols]
        y = df_clean['is_viral']
        
        return X, y
    
    def split_data(self, X, y, test_size: float = None):
        """
        Split data into train and test sets
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or config.TEST_SIZE
        
        print("\n" + "=" * 70)
        print("📊 SPLITTING DATA")
        print("=" * 70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
        )
        
        # Split info table
        split_table = [
            ["Training Set", len(X_train), f"{len(X_train)/len(X)*100:.1f}%"],
            ["Test Set", len(X_test), f"{len(X_test)/len(X)*100:.1f}%"],
            ["Total", len(X), "100.0%"]
        ]
        print("\n📊 Train/Test Split:")
        print(tabulate(split_table, headers=["Dataset", "Samples", "Percentage"], tablefmt="pretty"))
        
        return X_train, X_test, y_train, y_test
