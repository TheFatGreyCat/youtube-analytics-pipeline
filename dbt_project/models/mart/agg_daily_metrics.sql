{{
    config(
        materialized='table',
        partition_by={
            'field': 'metric_date',
            'data_type': 'date',
            'granularity': 'month'
        },
        cluster_by=['channel_id', 'metric_date']
    )
}}

with videos as (
    select * from {{ ref('int_videos__enhanced') }}
),

engagement as (
    select
        video_id,
        engagement_score,
        like_rate_pct
    from {{ ref('int_engagement_metrics') }}
),

daily_agg as (
    select
        date(v.published_at) as metric_date,
        v.channel_id,
        any_value(v.channel_name)  as channel_name,
        any_value(v.country_code)  as country_code,

        count(distinct v.video_id) as videos_published,
        sum(v.view_count)          as total_views,
        sum(v.like_count)          as total_likes,
        sum(v.comment_count)       as total_comments,
        avg(e.engagement_score)    as avg_engagement_score,
        avg(e.like_rate_pct)       as avg_like_rate_pct,
        max(v.view_count)          as max_video_views,

        current_timestamp() as dbt_updated_at

    from videos v
    left join engagement e on v.video_id = e.video_id
    group by 1, 2
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['metric_date', 'channel_id']) }} as date_channel_key,
        *
    from daily_agg
)

select * from final
