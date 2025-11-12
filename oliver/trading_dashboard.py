#!/usr/bin/env python3
"""
Real-time visualization dashboard for volatility trading logs using Streamlit.

Reads CSV log files and displays live visualizations of:
- Prediction accuracy and trends
- Volatility and spread
- BTC price movements
- Trading activity (fills)
- Model parameters

Usage:
    uv run streamlit run oliver/trading_dashboard.py
    uv run streamlit run oliver/trading_dashboard.py --server.port 8502
"""

import os
import sys
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Log file paths
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
CYCLES_LOG_FILE = os.path.join(LOGS_DIR, "volatility_trading_cycles.csv")
FILLS_LOG_FILE = os.path.join(LOGS_DIR, "volatility_trading_fills.csv")
PREDICTIONS_LOG_FILE = os.path.join(LOGS_DIR, "volatility_trading_predictions.csv")

# Page config
st.set_page_config(
    page_title="Volatility Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=1)  # Cache for 1 second to allow updates
def load_data():
    """Load all CSV log files."""
    data = {}
    
    # Load cycles log
    if os.path.exists(CYCLES_LOG_FILE):
        try:
            cycles_df = pd.read_csv(CYCLES_LOG_FILE)
            cycles_df['timestamp'] = pd.to_datetime(cycles_df['timestamp'])
            cycles_df = cycles_df.sort_values('timestamp')
            data['cycles'] = cycles_df
        except Exception as e:
            st.error(f"Could not load cycles log: {e}")
            data['cycles'] = pd.DataFrame()
    else:
        data['cycles'] = pd.DataFrame()
    
    # Load fills log
    if os.path.exists(FILLS_LOG_FILE):
        try:
            fills_df = pd.read_csv(FILLS_LOG_FILE)
            fills_df['timestamp'] = pd.to_datetime(fills_df['timestamp'])
            fills_df = fills_df.sort_values('timestamp')
            data['fills'] = fills_df
        except Exception as e:
            st.error(f"Could not load fills log: {e}")
            data['fills'] = pd.DataFrame()
    else:
        data['fills'] = pd.DataFrame()
    
    # Load predictions log
    if os.path.exists(PREDICTIONS_LOG_FILE):
        try:
            predictions_df = pd.read_csv(PREDICTIONS_LOG_FILE)
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            predictions_df = predictions_df.sort_values('timestamp')
            data['predictions'] = predictions_df
        except Exception as e:
            st.error(f"Could not load predictions log: {e}")
            data['predictions'] = pd.DataFrame()
    else:
        data['predictions'] = pd.DataFrame()
    
    return data


def create_price_chart(cycles_recent):
    """Create BTC price vs threshold chart."""
    fig = go.Figure()
    
    if not cycles_recent.empty:
        fig.add_trace(go.Scatter(
            x=cycles_recent['timestamp'],
            y=cycles_recent['btc_price_current'],
            name='BTC Price',
            line=dict(color='blue', width=2),
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            x=cycles_recent['timestamp'],
            y=cycles_recent['threshold_price'],
            name='Threshold',
            line=dict(color='red', width=1, dash='dash'),
            mode='lines'
        ))
    
    fig.update_layout(
        title='BTC Price vs Threshold',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    return fig


def create_probability_chart(cycles_recent):
    """Create predicted probability chart."""
    fig = go.Figure()
    
    if not cycles_recent.empty:
        fig.add_trace(go.Scatter(
            x=cycles_recent['timestamp'],
            y=cycles_recent['predicted_price'],
            name='Predicted Prob',
            line=dict(color='green', width=2),
            mode='lines'
        ))
        
        if 'market_price' in cycles_recent.columns:
            market_prices = pd.to_numeric(cycles_recent['market_price'], errors='coerce')
            valid_market = market_prices.notna()
            if valid_market.any():
                fig.add_trace(go.Scatter(
                    x=cycles_recent.loc[valid_market, 'timestamp'],
                    y=market_prices[valid_market],
                    name='Market Price',
                    line=dict(color='orange', width=1, dash='dot'),
                    mode='lines'
                ))
        
        fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Predicted Probability',
        xaxis_title='Time',
        yaxis_title='Probability',
        yaxis=dict(range=[0, 1]),
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    return fig


def create_volatility_chart(cycles_recent):
    """Create volatility chart."""
    fig = go.Figure()
    
    if not cycles_recent.empty:
        fig.add_trace(go.Scatter(
            x=cycles_recent['timestamp'],
            y=cycles_recent['volatility'],
            name='Volatility',
            line=dict(color='purple', width=2),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)'
        ))
    
    fig.update_layout(
        title='Volatility (σ per minute)',
        xaxis_title='Time',
        yaxis_title='Volatility',
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    return fig


def create_spread_chart(cycles_recent):
    """Create spread chart."""
    fig = go.Figure()
    
    if not cycles_recent.empty:
        fig.add_trace(go.Scatter(
            x=cycles_recent['timestamp'],
            y=cycles_recent['spread_cents'],
            name='Spread',
            line=dict(color='cyan', width=2),
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(0, 255, 255, 0.1)'
        ))
    
    fig.update_layout(
        title='Spread (cents)',
        xaxis_title='Time',
        yaxis_title='Spread (cents)',
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    return fig


def create_edge_chart(predictions_recent):
    """Create edge chart."""
    fig = go.Figure()
    
    if not predictions_recent.empty and 'edge' in predictions_recent.columns:
        edge_data = pd.to_numeric(predictions_recent['edge'], errors='coerce')
        valid_edge = edge_data.notna()
        if valid_edge.any():
            fig.add_trace(go.Scatter(
                x=predictions_recent.loc[valid_edge, 'timestamp'],
                y=edge_data[valid_edge],
                name='Edge',
                line=dict(color='red', width=2),
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title='Edge (Predicted - Market)',
        xaxis_title='Time',
        yaxis_title='Edge',
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    return fig


def create_parameters_chart(predictions_recent):
    """Create model parameters chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    if not predictions_recent.empty:
        if 'sigma_dt' in predictions_recent.columns:
            sigma_data = pd.to_numeric(predictions_recent['sigma_dt'], errors='coerce')
            valid_sigma = sigma_data.notna()
            if valid_sigma.any():
                fig.add_trace(
                    go.Scatter(
                        x=predictions_recent.loc[valid_sigma, 'timestamp'],
                        y=sigma_data[valid_sigma],
                        name='σ (volatility)',
                        line=dict(color='blue', width=2),
                        mode='lines'
                    ),
                    secondary_y=False
                )
        
        if 'lam' in predictions_recent.columns:
            lam_data = pd.to_numeric(predictions_recent['lam'], errors='coerce')
            valid_lam = lam_data.notna()
            if valid_lam.any():
                fig.add_trace(
                    go.Scatter(
                        x=predictions_recent.loc[valid_lam, 'timestamp'],
                        y=lam_data[valid_lam],
                        name='λ (jump intensity)',
                        line=dict(color='orange', width=2),
                        mode='lines'
                    ),
                    secondary_y=True
                )
    
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Volatility (σ)", secondary_y=False)
    fig.update_yaxes(title_text="Jump Intensity (λ)", secondary_y=True)
    fig.update_layout(
        title='Model Parameters',
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    return fig


def create_fills_chart(fills_recent):
    """Create fills scatter chart."""
    fig = go.Figure()
    
    if not fills_recent.empty:
        buy_fills = fills_recent[fills_recent['action'] == 'buy']
        sell_fills = fills_recent[fills_recent['action'] == 'sell']
        
        if not buy_fills.empty:
            fig.add_trace(go.Scatter(
                x=buy_fills['timestamp'],
                y=buy_fills['price'],
                name='Buy Fills',
                mode='markers',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        if not sell_fills.empty:
            fig.add_trace(go.Scatter(
                x=sell_fills['timestamp'],
                y=sell_fills['price'],
                name='Sell Fills',
                mode='markers',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
    
    fig.update_layout(
        title='Fill Prices Over Time',
        xaxis_title='Time',
        yaxis_title='Fill Price',
        yaxis=dict(range=[0, 1]),
        hovermode='closest',
        height=300,
        showlegend=True
    )
    return fig


def create_position_sizes_chart(cycles_recent):
    """Create position sizes chart."""
    fig = go.Figure()
    
    if not cycles_recent.empty:
        buy_sizes = pd.to_numeric(cycles_recent['buy_position_size'], errors='coerce')
        sell_sizes = pd.to_numeric(cycles_recent['sell_position_size'], errors='coerce')
        valid_buy = buy_sizes.notna()
        valid_sell = sell_sizes.notna()
        
        if valid_buy.any():
            fig.add_trace(go.Scatter(
                x=cycles_recent.loc[valid_buy, 'timestamp'],
                y=buy_sizes[valid_buy],
                name='Buy Size',
                line=dict(color='green', width=2),
                mode='lines'
            ))
        
        if valid_sell.any():
            fig.add_trace(go.Scatter(
                x=cycles_recent.loc[valid_sell, 'timestamp'],
                y=sell_sizes[valid_sell],
                name='Sell Size',
                line=dict(color='red', width=2),
                mode='lines'
            ))
    
    fig.update_layout(
        title='Position Sizes',
        xaxis_title='Time',
        yaxis_title='Contracts',
        hovermode='x unified',
        height=300,
        showlegend=True
    )
    return fig


def main():
    """Main Streamlit app."""
    st.title("📈 Volatility Trading Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        time_window_hours = st.slider(
            "Time Window (hours)",
            min_value=1.0,
            max_value=24.0,
            value=6.0,
            step=0.5
        )
        
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider(
            "Refresh interval (seconds)",
            min_value=1,
            max_value=60,
            value=2,
            step=1,
            disabled=not auto_refresh
        )
        
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.header("📊 Statistics")
        
        # Load data
        data = load_data()
        cycles_df = data['cycles']
        fills_df = data['fills']
        predictions_df = data['predictions']
        
        # Calculate statistics
        if not cycles_df.empty:
            st.metric("Total Cycles", len(cycles_df))
            if 'predicted_price' in cycles_df.columns:
                avg_pred = cycles_df['predicted_price'].mean()
                st.metric("Avg Prediction", f"{avg_pred:.3f}")
            if 'volatility' in cycles_df.columns:
                avg_vol = cycles_df['volatility'].mean()
                st.metric("Avg Volatility", f"{avg_vol:.6f}")
        
        if not fills_df.empty:
            total_fills = len(fills_df)
            buy_fills_count = len(fills_df[fills_df['action'] == 'buy'])
            sell_fills_count = len(fills_df[fills_df['action'] == 'sell'])
            st.metric("Total Fills", total_fills)
            st.metric("Buy Fills", buy_fills_count)
            st.metric("Sell Fills", sell_fills_count)
        
        if not predictions_df.empty:
            st.metric("Total Predictions", len(predictions_df))
    
    # Filter data by time window
    time_window = pd.Timedelta(hours=time_window_hours)
    
    if not cycles_df.empty:
        latest_time = cycles_df['timestamp'].max()
        start_time = max(latest_time - time_window, cycles_df['timestamp'].min())
        cycles_recent = cycles_df[cycles_df['timestamp'] >= start_time]
    else:
        cycles_recent = cycles_df
    
    if not predictions_df.empty:
        latest_time_pred = predictions_df['timestamp'].max()
        start_time_pred = max(latest_time_pred - time_window, predictions_df['timestamp'].min())
        predictions_recent = predictions_df[predictions_df['timestamp'] >= start_time_pred]
    else:
        predictions_recent = predictions_df
    
    if not fills_df.empty:
        latest_time_fills = fills_df['timestamp'].max()
        start_time_fills = max(latest_time_fills - time_window, fills_df['timestamp'].min())
        fills_recent = fills_df[fills_df['timestamp'] >= start_time_fills]
    else:
        fills_recent = fills_df
    
    # Check if we have data
    has_data = (not cycles_df.empty or not fills_df.empty or not predictions_df.empty)
    
    if not has_data:
        st.warning("⚠️ No data found. Waiting for log files to be created...")
        st.info(f"Looking for log files in: `{LOGS_DIR}`")
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
        return
    
    # Main dashboard
    # Row 1: Price, Probability, Volatility
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(create_price_chart(cycles_recent), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_probability_chart(cycles_recent), use_container_width=True)
    
    with col3:
        st.plotly_chart(create_volatility_chart(cycles_recent), use_container_width=True)
    
    # Row 2: Spread, Edge, Parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.plotly_chart(create_spread_chart(cycles_recent), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_edge_chart(predictions_recent), use_container_width=True)
    
    with col3:
        st.plotly_chart(create_parameters_chart(predictions_recent), use_container_width=True)
    
    # Row 3: Fills, Position Sizes
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_fills_chart(fills_recent), use_container_width=True)
    
    with col2:
        st.plotly_chart(create_position_sizes_chart(cycles_recent), use_container_width=True)
    
    # Data tables
    with st.expander("📋 Recent Cycles Data"):
        if not cycles_recent.empty:
            st.dataframe(
                cycles_recent.tail(100)[['timestamp', 'cycle_num', 'btc_price_current', 
                                        'predicted_price', 'volatility', 'spread_cents', 
                                        'buy_position_size', 'sell_position_size', 'status']],
                use_container_width=True
            )
        else:
            st.info("No cycles data available")
    
    with st.expander("📋 Recent Fills Data"):
        if not fills_recent.empty:
            st.dataframe(
                fills_recent.tail(100)[['timestamp', 'ticker', 'action', 'side', 
                                       'count', 'price']],
                use_container_width=True
            )
        else:
            st.info("No fills data available")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
