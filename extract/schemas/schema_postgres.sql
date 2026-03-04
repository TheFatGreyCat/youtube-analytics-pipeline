-- POSTGRESQL SCHEMA - Metadata & Orchestration
-- Purpose: Crawl scheduling, execution logs, API quota tracking
-- Data Storage: All YouTube data goes to BigQuery

CREATE TABLE IF NOT EXISTS channels_config (
    channel_id VARCHAR(100) PRIMARY KEY,
    channel_name VARCHAR(255) NOT NULL,
    crawl_frequency_hours INT DEFAULT 24,
    crawl_status VARCHAR(50) DEFAULT 'pending',
    last_crawl_ts TIMESTAMP,
    next_crawl_ts TIMESTAMP DEFAULT NOW(),
    last_video_published_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    priority INT DEFAULT 1,
    include_comments BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_channels_next_crawl ON channels_config(next_crawl_ts) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_channels_status ON channels_config(crawl_status);
CREATE INDEX IF NOT EXISTS idx_channels_priority ON channels_config(priority DESC, next_crawl_ts);

-- CRAWL EXECUTION LOG
CREATE TABLE IF NOT EXISTS crawl_log (
    id SERIAL PRIMARY KEY,
    channel_id VARCHAR(100) NOT NULL,
    crawl_ts TIMESTAMP DEFAULT NOW(),
    records_fetched INT DEFAULT 0,
    status VARCHAR(50) NOT NULL,
    error_msg TEXT,
    execution_time_seconds FLOAT,
    api_quota_used INT DEFAULT 0,
    CONSTRAINT fk_crawl_channel FOREIGN KEY (channel_id) 
        REFERENCES channels_config(channel_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_crawl_log_channel ON crawl_log(channel_id, crawl_ts DESC);
CREATE INDEX IF NOT EXISTS idx_crawl_log_status ON crawl_log(status, crawl_ts DESC);
CREATE INDEX IF NOT EXISTS idx_crawl_log_ts ON crawl_log(crawl_ts DESC);

-- API QUOTA TRACKING
CREATE TABLE IF NOT EXISTS api_quota_usage (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE UNIQUE,
    quota_used INT DEFAULT 0,
    daily_limit INT DEFAULT 10000,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quota_date ON api_quota_usage(date DESC);

-- UTILITY FUNCTIONS
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_channels_config_updated ON channels_config;

CREATE TRIGGER trg_channels_config_updated
BEFORE UPDATE ON channels_config
FOR EACH ROW EXECUTE FUNCTION update_updated_at();

INSERT INTO api_quota_usage (date, quota_used, daily_limit)
VALUES (CURRENT_DATE, 0, 10000)
ON CONFLICT (date) DO NOTHING;
