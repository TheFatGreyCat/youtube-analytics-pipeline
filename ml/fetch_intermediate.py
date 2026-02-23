"""
Fetch data from Intermediate Layer (BigQuery)
Includes: int_videos__enhanced, int_engagement_metrics, int_channel_summary
"""
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from extract.db_manager import BigQueryManager
from extract.config import GCP_PROJECT_ID, BQ_DATASET_ID


class IntermediateLayerFetcher:
    """Fetch data from intermediate layer tables"""
    
    def __init__(self):
        self.bq_manager = BigQueryManager()
        self.client = self.bq_manager.client
        self.project = self.bq_manager.project_id
        self.dataset = self.bq_manager.dataset_id
        
        print(f"✅ Connected to BigQuery")
        print(f"   Project: {self.project}")
        print(f"   Default Dataset: {self.dataset}")
        print(f"   📊 Querying from: intermediate schema")
    
    def query_to_dataframe(self, sql: str) -> pd.DataFrame:
        """Execute SQL and return DataFrame"""
        try:
            print(f"\n🔍 Executing query...")
            query_job = self.client.query(sql)
            results = query_job.result()
            df = results.to_dataframe()
            print(f"✅ Retrieved {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
            raise
    
    def get_videos_enhanced(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch enhanced video data with joined channel info and calculated metrics
        
        Args:
            limit: Maximum rows to return
            
        Returns:
            pd.DataFrame: Enhanced video data
        """
        sql = f"""
        SELECT *
        FROM `{self.project}.intermediate.int_videos__enhanced`
        ORDER BY view_count DESC
        LIMIT {limit}
        """
        return self.query_to_dataframe(sql)
    
    def get_engagement_metrics(self, limit: int = 1000, min_views: int = None) -> pd.DataFrame:
        """
        Fetch engagement metrics (like_rate, comment_rate, engagement_score, etc.)
        
        Args:
            limit: Maximum rows to return
            min_views: Filter videos with minimum view count
            
        Returns:
            pd.DataFrame: Engagement metrics
        """
        where_clause = f"WHERE view_count >= {min_views}" if min_views else ""
        
        sql = f"""
        SELECT *
        FROM `{self.project}.intermediate.int_engagement_metrics`
        {where_clause}
        ORDER BY engagement_score DESC NULLS LAST
        LIMIT {limit}
        """
        return self.query_to_dataframe(sql)
    
    def get_channel_summary(self) -> pd.DataFrame:
        """
        Fetch channel summary with aggregated metrics
        
        Returns:
            pd.DataFrame: Channel summary
        """
        sql = f"""
        SELECT *
        FROM `{self.project}.intermediate.int_channel_summary`
        ORDER BY total_views DESC
        """
        return self.query_to_dataframe(sql)
    
    def get_all_intermediate_data(self, limit: int = 10000) -> pd.DataFrame:
        """
        Fetch ALL data from intermediate layer (join all 3 tables)
        Combines: int_videos__enhanced + int_engagement_metrics + int_channel_summary
        
        Args:
            limit: Maximum rows to return
            
        Returns:
            pd.DataFrame: Complete dataset with all features
        """
        sql = f"""
        SELECT 
            e.*,
            v.title,
            v.description,
            v.tags,
            v.category_id,
            v.published_at,
            v.duration_seconds,
            c.total_views as channel_total_views,
            c.total_videos_crawled as channel_total_videos,
            c.avg_views_per_video as channel_avg_views,
            c.avg_like_rate_pct as channel_avg_like_rate,
            c.avg_comment_rate_pct as channel_avg_comment_rate
        FROM `{self.project}.intermediate.int_engagement_metrics` e
        LEFT JOIN `{self.project}.intermediate.int_videos__enhanced` v
            ON e.video_id = v.video_id
        LEFT JOIN `{self.project}.intermediate.int_channel_summary` c
            ON e.channel_id = c.channel_id
        ORDER BY e.engagement_score DESC NULLS LAST
        LIMIT {limit}
        """
        return self.query_to_dataframe(sql)
    
    def get_high_engagement_videos(self, top_n: int = 100, min_views: int = 1000) -> pd.DataFrame:
        """
        Get top high-engagement videos
        
        Args:
            top_n: Number of top videos to return
            min_views: Minimum view count threshold
            
        Returns:
            pd.DataFrame: Top engagement videos
        """
        sql = f"""
        SELECT 
            video_id,
            title,
            channel_name,
            view_count,
            like_count,
            comment_count,
            like_rate_pct,
            comment_rate_pct,
            engagement_score,
            published_at
        FROM `{self.project}.intermediate.int_engagement_metrics`
        WHERE view_count >= {min_views}
          AND engagement_score IS NOT NULL
        ORDER BY engagement_score DESC
        LIMIT {top_n}
        """
        return self.query_to_dataframe(sql)
    
    def get_videos_by_channel(self, channel_id: str = None, channel_name: str = None, limit: int = 1000) -> pd.DataFrame:
        """
        Get videos filtered by channel (with engagement metrics for ML predictions)
        
        Args:
            channel_id: Filter by channel ID
            channel_name: Filter by channel name (partial match)
            limit: Maximum rows
            
        Returns:
            pd.DataFrame: Videos from specified channel with all engagement features
        """
        where_clauses = []
        
        if channel_id:
            where_clauses.append(f"channel_id = '{channel_id}'")
        
        if channel_name:
            where_clauses.append(f"channel_name LIKE '%{channel_name}%'")
        
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        # Use int_engagement_metrics which has all features needed for ML
        sql = f"""
        SELECT *
        FROM `{self.project}.intermediate.int_engagement_metrics`
        {where_sql}
        ORDER BY published_at DESC
        LIMIT {limit}
        """
        return self.query_to_dataframe(sql)
    
    def get_video_statistics_summary(self) -> pd.DataFrame:
        """
        Get overall statistics summary from intermediate layer
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        sql = f"""
        SELECT 
            COUNT(DISTINCT video_id) as total_videos,
            COUNT(DISTINCT channel_id) as total_channels,
            SUM(view_count) as total_views,
            SUM(like_count) as total_likes,
            SUM(comment_count) as total_comments,
            AVG(view_count) as avg_views_per_video,
            AVG(like_rate_pct) as avg_like_rate_pct,
            AVG(comment_rate_pct) as avg_comment_rate_pct,
            AVG(engagement_score) as avg_engagement_score,
            MAX(view_count) as max_views,
            MIN(published_at) as earliest_video,
            MAX(published_at) as latest_video
        FROM `{self.project}.intermediate.int_engagement_metrics`
        """
        return self.query_to_dataframe(sql)
    
    def export_to_csv(self, df: pd.DataFrame, filename: str, output_dir: str = "data") -> str:
        """Export DataFrame to CSV"""
        from datetime import datetime
        output_path = Path(__file__).parent.parent / output_dir
        output_path.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = output_path / f"{filename}_{timestamp}.csv"
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"✅ Exported to: {filepath}")
        return str(filepath)


def main():
    """Display data statistics"""
    fetcher = IntermediateLayerFetcher()
    
    print("\n📊 THỐNG KÊ DỮ LIỆU")
    print("=" * 50)
    
    df = fetcher.get_video_statistics_summary()
    
    print(f"📹 Tổng số video:      {int(df['total_videos'].iloc[0]):,}")
    print(f"📺 Tổng số channel:    {int(df['total_channels'].iloc[0]):,}")
    print(f"👁️  Tổng số view:       {int(df['total_views'].iloc[0]):,}")
    print(f"👍 Tổng số like:       {int(df['total_likes'].iloc[0]):,}")
    print(f"💬 Tổng số comment:    {int(df['total_comments'].iloc[0]):,}")
    print(f"📈 TB view/video:      {int(df['avg_views_per_video'].iloc[0]):,}")
    print(f"💯 TB engagement:      {df['avg_engagement_score'].iloc[0]:.2f}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
