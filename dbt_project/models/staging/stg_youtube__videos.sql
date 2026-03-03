{{
    config(
        materialized='view'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'raw_videos') }}
),

valid_channel_ids as (
    select id as channel_id
    from {{ source('youtube_raw', 'raw_channels') }}
    where ARRAY_LENGTH(JSON_QUERY_ARRAY(raw, '$.items')) > 0
),

flattened as (
    select
        -- Primary Key
        id as video_id,

        -- Snippet
        JSON_VALUE(raw, '$.snippet.channelId')       as channel_id,
        JSON_VALUE(raw, '$.snippet.title')           as title,
        JSON_VALUE(raw, '$.snippet.description')     as description,
        JSON_QUERY(raw, '$.snippet.tags')            as tags,
        JSON_VALUE(raw, '$.snippet.categoryId')      as category_id,
        JSON_VALUE(raw, '$.snippet.defaultLanguage') as default_language,
        TIMESTAMP(JSON_VALUE(raw, '$.snippet.publishedAt')) as published_at,

        -- Statistics
        CAST(JSON_VALUE(raw, '$.statistics.viewCount')    as INT64)              as view_count,
        COALESCE(CAST(JSON_VALUE(raw, '$.statistics.likeCount')    as INT64), 0) as like_count,
        COALESCE(CAST(JSON_VALUE(raw, '$.statistics.commentCount') as INT64), 0) as comment_count,

        -- Content Details
        JSON_VALUE(raw, '$.contentDetails.duration')                as duration_iso8601,
        CAST(JSON_VALUE(raw, '$.contentDetails.caption') as BOOL)   as has_caption,
        JSON_VALUE(raw, '$.contentDetails.definition')              as definition,

        -- Status
        COALESCE(JSON_VALUE(raw, '$.status.privacyStatus'), 'public') as privacy_status,
        CAST(JSON_VALUE(raw, '$.status.embeddable')  as BOOL) as is_embeddable,
        CAST(JSON_VALUE(raw, '$.status.madeForKids') as BOOL) as is_made_for_kids,

        -- Metadata
        CAST(ingestion_time as TIMESTAMP) as crawled_at,
        CURRENT_TIMESTAMP()               as dbt_loaded_at

        {{ get_passthrough_columns('youtube__video_passthrough_columns') }}

    from source
    where COALESCE(JSON_VALUE(raw, '$.status.privacyStatus'), 'public') = 'public'
        and JSON_VALUE(raw, '$.contentDetails.duration') is not null
)

select f.*
from flattened f
inner join valid_channel_ids vc on f.channel_id = vc.channel_id
