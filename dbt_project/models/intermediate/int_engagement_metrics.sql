{{
    config(
        materialized='view'
    )
}}

with videos as (
    select * from {{ ref('int_videos__enhanced') }}
),

metrics as (
    select
        video_id,
        channel_id,
        title,
        published_at,
        published_date,
        view_count,
        like_count,
        comment_count,
        days_since_published,
        video_length_category,
        channel_name,
        channel_subscribers,
        country_code,
        
        case 
            when view_count > 0 then least(like_count / view_count * 100, 100.0)
            else 0 
        end as like_rate_pct,
        
        case 
            when view_count > 0 then least(comment_count / view_count * 100, 100.0)
            else 0 
        end as comment_rate_pct,
        
        case 
            when view_count > 0 then (like_count + comment_count * 2) / view_count * 100 
            else 0 
        end as engagement_score,
        
        view_count / nullif(days_since_published, 0) as avg_views_per_day,
        
        case
            when view_count = 0 then 'low'
            when like_count / view_count >= 0.05 then 'high'
            when like_count / view_count >= 0.02 then 'medium'
            else 'low'
        end as engagement_level,
        
        case
            when view_count / nullif(days_since_published, 0) > 
                 channel_subscribers * 0.1 then true
            else false
        end as is_potentially_viral
        
    from videos
    where days_since_published > 0
)

select * from metrics
