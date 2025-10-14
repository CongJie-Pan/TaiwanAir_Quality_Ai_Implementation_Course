"""
Visualization Functions for Air Quality Streamlit App

This module provides Plotly-based visualization functions for the Streamlit application:
- Time series plots with AQI thresholds
- Heatmaps for crosstab analysis
- Geographic distribution maps
- Statistical charts (histograms, box plots, bar charts)
- Correlation matrices
- KPI metric displays

Author: Claude Code
Date: 2025-10-14
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


def create_time_series_plot(
    df: pd.DataFrame,
    y_column: str = 'aqi',
    title: Optional[str] = None,
    show_thresholds: bool = True
) -> go.Figure:
    """
    Create time series line plot with optional threshold lines.

    Args:
        df: DataFrame with time series data
        y_column: Column name for y-axis values
        title: Plot title (auto-generated if None)
        show_thresholds: Whether to show AQI threshold lines

    Returns:
        Plotly Figure object

    Example:
        >>> fig = create_time_series_plot(df, 'aqi', '台北市AQI趨勢')
        >>> st.plotly_chart(fig)
    """
    # Sort by date
    df_sorted = df.sort_values('date')

    # Create figure
    fig = go.Figure()

    # Add main line
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted[y_column],
        mode='lines',
        name=y_column.upper(),
        line=dict(color='#1f77b4', width=2)
    ))

    # Add threshold lines for AQI
    if show_thresholds and y_column == 'aqi':
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="green",
            annotation_text="良好(50)",
            annotation_position="right"
        )
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="yellow",
            annotation_text="普通(100)",
            annotation_position="right"
        )
        fig.add_hline(
            y=150,
            line_dash="dash",
            line_color="orange",
            annotation_text="不健康(150)",
            annotation_position="right"
        )

    # Update layout
    fig.update_layout(
        title=title or f"{y_column.upper()} 時間序列趨勢",
        xaxis_title="日期",
        yaxis_title=y_column.upper(),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_multi_series_plot(
    df: pd.DataFrame,
    y_columns: List[str],
    title: str = "多指標時間序列比較"
) -> go.Figure:
    """
    Create time series plot with multiple y-axis series.

    Args:
        df: DataFrame with time series data
        y_columns: List of column names to plot
        title: Plot title

    Returns:
        Plotly Figure object
    """
    df_sorted = df.sort_values('date')

    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for i, col in enumerate(y_columns):
        fig.add_trace(go.Scatter(
            x=df_sorted['date'],
            y=df_sorted[col],
            mode='lines',
            name=col.upper(),
            line=dict(color=colors[i % len(colors)], width=2)
        ))

    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title="數值",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def create_crosstab_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    agg_func: str = 'mean',
    title: Optional[str] = None
) -> go.Figure:
    """
    Create heatmap from crosstab analysis.

    Args:
        df: Input DataFrame
        x_col: Column for x-axis (columns)
        y_col: Column for y-axis (rows)
        value_col: Column for values
        agg_func: Aggregation function ('mean', 'sum', 'count', etc.)
        title: Plot title

    Returns:
        Plotly Figure object

    Example:
        >>> fig = create_crosstab_heatmap(df, 'month', 'county', 'aqi', 'mean')
    """
    # Create pivot table
    pivot_data = df.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc=agg_func
    )

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        text=pivot_data.values.round(1),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=value_col.upper())
    ))

    fig.update_layout(
        title=title or f"{y_col} × {x_col} 熱力圖",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template='plotly_white',
        height=500,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial, Microsoft YaHei, sans-serif",
            font_color="black"
        )
    )

    return fig


def create_map_plot(
    df: pd.DataFrame,
    size_col: str = 'pm2.5',
    color_col: str = 'aqi',
    title: str = "監測站地理分布"
) -> go.Figure:
    """
    Create geographic scatter map of monitoring stations.

    Args:
        df: DataFrame with station data
        size_col: Column for bubble size
        color_col: Column for bubble color
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Aggregate by station
    station_data = df.groupby(['sitename', 'county', 'latitude', 'longitude']).agg({
        'aqi': 'mean',
        'pm2.5': 'mean',
        'pm10': 'mean'
    }).reset_index()

    fig = px.scatter_mapbox(
        station_data,
        lat='latitude',
        lon='longitude',
        size=size_col,
        color=color_col,
        hover_name='sitename',
        hover_data={
            'county': True,
            'aqi': ':.1f',
            'pm2.5': ':.1f',
            'latitude': False,
            'longitude': False
        },
        color_continuous_scale='RdYlGn_r',
        zoom=7,
        center={"lat": 23.5, "lon": 121},
        title=title
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        height=600
    )

    return fig


def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
    plot_type: str = 'histogram',
    title: Optional[str] = None
) -> go.Figure:
    """
    Create distribution plot (histogram or box plot).

    Args:
        df: Input DataFrame
        column: Column name to plot
        plot_type: 'histogram' or 'box'
        title: Plot title

    Returns:
        Plotly Figure object
    """
    if plot_type == 'histogram':
        fig = px.histogram(
            df,
            x=column,
            nbins=50,
            title=title or f"{column} 分布直方圖",
            labels={column: column.upper()},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title=column.upper(),
            yaxis_title="頻次",
            template='plotly_white',
            height=400
        )

    elif plot_type == 'box':
        fig = px.box(
            df,
            y=column,
            title=title or f"{column} 箱型圖",
            labels={column: column.upper()},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            yaxis_title=column.upper(),
            template='plotly_white',
            height=400
        )

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'histogram' or 'box'.")

    return fig


def create_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    orientation: str = 'v',
    title: Optional[str] = None,
    color_col: Optional[str] = None
) -> go.Figure:
    """
    Create bar chart.

    Args:
        df: Input DataFrame
        x_col: Column for x-axis
        y_col: Column for y-axis (values)
        orientation: 'v' (vertical) or 'h' (horizontal)
        title: Plot title
        color_col: Optional column for color grouping

    Returns:
        Plotly Figure object
    """
    if color_col:
        fig = px.bar(
            df,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            color=color_col,
            orientation=orientation,
            title=title or f"{y_col} by {x_col}",
            barmode='group'
        )
    else:
        fig = px.bar(
            df,
            x=x_col if orientation == 'v' else y_col,
            y=y_col if orientation == 'v' else x_col,
            orientation=orientation,
            title=title or f"{y_col} by {x_col}"
        )

    fig.update_layout(
        template='plotly_white',
        height=400,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial, Microsoft YaHei, sans-serif",
            font_color="black"
        )
    )

    # Add hover template for better text visibility
    if orientation == 'v':
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>%{y:.1f}<extra></extra>'
        )
    else:  # horizontal
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>%{x:.1f}<extra></extra>'
        )

    return fig


def create_correlation_matrix(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "污染物相關性矩陣"
) -> go.Figure:
    """
    Create correlation matrix heatmap.

    Args:
        df: Input DataFrame
        columns: List of columns to include in correlation
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="相關係數")
    ))

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        template='plotly_white',
        height=500,
        width=500
    )

    return fig


def create_seasonal_pattern_plot(
    df: pd.DataFrame,
    value_col: str = 'aqi',
    title: str = "季節性模式分析"
) -> go.Figure:
    """
    Create seasonal pattern analysis plot.

    Args:
        df: DataFrame with 'season' column
        value_col: Column to analyze
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Calculate seasonal statistics
    seasonal_stats = df.groupby('season')[value_col].agg(['mean', 'std']).reset_index()

    fig = go.Figure()

    # Add bar for mean
    fig.add_trace(go.Bar(
        x=seasonal_stats['season'],
        y=seasonal_stats['mean'],
        name='平均值',
        marker_color='lightblue',
        error_y=dict(type='data', array=seasonal_stats['std']),
        hovertemplate='<b>季節: %{x}</b><br>平均值: %{y:.1f}<br>標準差: %{error_y.array:.1f}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="季節",
        yaxis_title=f"{value_col.upper()} 平均值",
        template='plotly_white',
        height=400,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial, Microsoft YaHei, sans-serif",
            font_color="black"
        )
    )

    return fig


def create_wind_rose(
    df: pd.DataFrame,
    title: str = "風向分布圖"
) -> go.Figure:
    """
    Create wind rose plot showing wind direction distribution.

    Args:
        df: DataFrame with 'winddirec' column
        title: Plot title

    Returns:
        Plotly Figure object
    """
    # Bin wind directions into 8 compass directions
    def direction_category(degree):
        if pd.isna(degree):
            return None
        if degree < 22.5 or degree >= 337.5:
            return '北'
        elif degree < 67.5:
            return '東北'
        elif degree < 112.5:
            return '東'
        elif degree < 157.5:
            return '東南'
        elif degree < 202.5:
            return '南'
        elif degree < 247.5:
            return '西南'
        elif degree < 292.5:
            return '西'
        else:
            return '西北'

    df_wind = df.copy()
    df_wind['wind_direction'] = df_wind['winddirec'].apply(direction_category)

    # Count occurrences
    wind_counts = df_wind['wind_direction'].value_counts()

    # Create polar bar chart
    fig = go.Figure(go.Barpolar(
        r=wind_counts.values,
        theta=wind_counts.index,
        marker_color=px.colors.sequential.Plasma_r,
        marker_line_color="black",
        marker_line_width=1,
        opacity=0.8,
        hovertemplate='<b>風向: %{theta}</b><br>次數: %{r}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True),
            angularaxis=dict(direction="clockwise")
        ),
        template='plotly_white',
        height=500,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial, Microsoft YaHei, sans-serif",
            font_color="black"
        )
    )

    return fig


def create_comparison_plot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str = "分組比較"
) -> go.Figure:
    """
    Create box plot for comparing groups.

    Args:
        df: Input DataFrame
        group_col: Column for grouping
        value_col: Column for values
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = px.box(
        df,
        x=group_col,
        y=value_col,
        title=title,
        color=group_col,
        labels={group_col: group_col, value_col: value_col.upper()}
    )

    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def create_trend_with_moving_average(
    df: pd.DataFrame,
    value_col: str = 'aqi',
    window: int = 7,
    title: str = "趨勢分析（含移動平均）"
) -> go.Figure:
    """
    Create time series plot with moving average.

    Args:
        df: DataFrame with time series data
        value_col: Column to plot
        window: Window size for moving average
        title: Plot title

    Returns:
        Plotly Figure object
    """
    df_sorted = df.sort_values('date')

    # Calculate moving average
    df_sorted[f'{value_col}_ma'] = df_sorted[value_col].rolling(window=window).mean()

    fig = go.Figure()

    # Add original data
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted[value_col],
        mode='lines',
        name='原始數據',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))

    # Add moving average
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted[f'{value_col}_ma'],
        mode='lines',
        name=f'{window}日移動平均',
        line=dict(color='#1f77b4', width=3)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="日期",
        yaxis_title=value_col.upper(),
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig
