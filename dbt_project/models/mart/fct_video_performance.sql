{{
    config(
        materialized='table',
        partition_by={
            'field': 'published_date',
            'data_type': 'date',
            'granularity': 'month'
        },
        cluster_by=['channel_id', 'video_id']
    )
}}

with metrics as (
    select * from {{ ref('int_engagement_metrics') }}
),

videos as (
    select * from {{ ref('int_videos__enhanced') }}
),

final as (
    select
        v.video_id,
        v.channel_id,
        v.category_id,
        v.title,
        v.published_at,
        v.published_date,
        v.published_year,
        v.published_month,
        v.channel_name,
        v.country_code,
        v.video_length_category,
        v.duration_seconds,
        
        v.view_count,
        v.like_count,
        v.comment_count,
        m.like_rate_pct,
        m.comment_rate_pct,
        m.engagement_score,
        m.avg_views_per_day,
        m.engagement_level,
        m.is_potentially_viral,
        
        v.has_caption,
        v.is_embeddable,
        v.is_made_for_kids,
        v.definition,
        
        v.crawled_at,
        current_timestamp() as dbt_updated_at
        
    from videos v
    left join metrics m on v.video_id = m.video_id
)

select * from final
