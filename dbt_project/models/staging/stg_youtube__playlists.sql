{{
    config(
        materialized='view'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'raw_playlists') }}
),

valid_channel_ids as (
    select id as channel_id
    from {{ source('youtube_raw', 'raw_channels') }}
    where ARRAY_LENGTH(JSON_QUERY_ARRAY(raw, '$.items')) > 0
),

flattened as (
    select
        -- Primary Key
        id as playlist_id,
        channel_id,

        -- Snippet
        JSON_VALUE(raw, '$.snippet.title')       as playlist_name,
        JSON_VALUE(raw, '$.snippet.description') as description,
        TIMESTAMP(JSON_VALUE(raw, '$.snippet.publishedAt')) as created_at,

        -- Content Details
        CAST(JSON_VALUE(raw, '$.contentDetails.itemCount') as INT64) as item_count,

        -- Status
        JSON_VALUE(raw, '$.status.privacyStatus') as privacy_status,

        -- Metadata
        CAST(ingestion_time as TIMESTAMP) as crawled_at,
        CURRENT_TIMESTAMP() as dbt_loaded_at

    from source
    where JSON_VALUE(raw, '$.status.privacyStatus') = 'public'
)

select f.*
from flattened f
inner join valid_channel_ids vc on f.channel_id = vc.channel_id
