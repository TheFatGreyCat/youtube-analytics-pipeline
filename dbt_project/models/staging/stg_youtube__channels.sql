{{
    config(
        materialized='view'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'raw_channels') }}
),

flattened as (
    select
        id as channel_id,
        
        JSON_VALUE(raw, '$.items[0].snippet.title') as channel_name,
        JSON_VALUE(raw, '$.items[0].snippet.description') as description,
        TIMESTAMP(JSON_VALUE(raw, '$.items[0].snippet.publishedAt')) as channel_created_at,
        JSON_VALUE(raw, '$.items[0].snippet.country') as country_code,
        JSON_VALUE(raw, '$.items[0].snippet.customUrl') as custom_url,
        
        CAST(JSON_VALUE(raw, '$.items[0].statistics.subscriberCount') as INT64) as subscriber_count,
        CAST(JSON_VALUE(raw, '$.items[0].statistics.viewCount') as INT64) as total_view_count,
        CAST(JSON_VALUE(raw, '$.items[0].statistics.videoCount') as INT64) as video_count,
        CAST(JSON_VALUE(raw, '$.items[0].statistics.hiddenSubscriberCount') as BOOL) as has_hidden_subscribers,
        
        JSON_VALUE(raw, '$.items[0].contentDetails.relatedPlaylists.uploads') as uploads_playlist_id,
        
        JSON_QUERY(raw, '$.items[0].topicDetails.topicIds') as topic_ids,
        
        CAST(ingestion_time as TIMESTAMP) as crawled_at,
        CURRENT_TIMESTAMP() as dbt_loaded_at
        
        {{ get_passthrough_columns('youtube__channel_passthrough_columns') }}
        
    from source
    where ARRAY_LENGTH(JSON_QUERY_ARRAY(raw, '$.items')) > 0
)

select * from flattened
