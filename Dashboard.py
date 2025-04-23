import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image

# Set page layout
st.set_page_config(layout="wide", page_title="ETF Portfolio Optimization Dashboard")

# Define sectors and tabs
sectors = ['Energy', 'Financials', 'Healthcare', 'Real Estate', 'Technology']
tab1, tab2 = st.tabs(["\U0001F4CA ETF Dashboard", "\U0001F50D Data Pipeline & Sentiment Analysis"])

# Dummy ML metrics
ml_metrics = {
    "Logistic Regression": {
        "Confusion": [[45, 10], [8, 37]],
        "Accuracy": 0.82,
        "Precision": 0.79,
        "AUC": 0.88
    },
    "Random Forest": {
        "Confusion": [[48, 7], [6, 39]],
        "Accuracy": 0.87,
        "Precision": 0.85,
        "AUC": 0.91
    }
}

# Helper to calculate TPR and FPR (unchanged)
def calculate_tpr_fpr(conf_matrix):
    tn, fp, fn, tp = np.array(conf_matrix).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    return tpr, fpr

# Function to load and process data (unchanged)
def load_data(start_date, end_date):
    # Load the ETF data
    try:
        etf_df = pd.read_csv("/Users/nilaysinghsolanki/Downloads/all_etf_data (1).csv")
        etf_df['Date'] = pd.to_datetime(etf_df['Date'], format='%d-%m-%Y', errors='coerce')
    except FileNotFoundError:
        raise FileNotFoundError("ETF data file not found at /Users/nilaysinghsolanki/Downloads/all_etf_data (1).csv")

    # Validate ETF data
    if etf_df.empty or etf_df['Date'].isna().all():
        raise ValueError("ETF data is empty or all dates are invalid")
    if 'Ticker' not in etf_df.columns or 'Close' not in etf_df.columns or 'Date' not in etf_df.columns:
        raise ValueError("ETF data must contain 'Date', 'Ticker', and 'Close' columns")

    # Convert 'Close' to numeric
    etf_df['Close'] = pd.to_numeric(etf_df['Close'], errors='coerce')

    # Drop rows with NaN dates or Close values
    etf_df = etf_df.dropna(subset=['Date', 'Close'])

    # Log ETF data for debugging
    st.write(f"ETF DataFrame shape after loading and cleaning: {etf_df.shape}")
    st.write(f"ETF Date range: {etf_df['Date'].min()} to {etf_df['Date'].max()}")

    # Filter by date range
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    etf_df = etf_df[(etf_df['Date'] >= start_date) & (etf_df['Date'] <= end_date)]

    # Validate ETF data after filtering
    if etf_df.empty:
        raise ValueError(f"ETF data is empty after filtering by date range {start_date} to {end_date}")

    # Normalize ETF data by ticker
    etf_df['Normalized'] = etf_df.groupby('Ticker')['Close'].transform(
        lambda x: x / x.iloc[0] if not x.empty and x.iloc[0] != 0 else 1
    )

    # Load the S&P 500 data
    try:
        sp500_df = pd.read_csv("/Users/nilaysinghsolanki/Downloads/S&P 500 Historical Data.csv")
        sp500_df['Date'] = pd.to_datetime(sp500_df['Date'], format='%m/%d/%Y', errors='coerce')
    except FileNotFoundError:
        raise FileNotFoundError("S&P 500 data file not found at /Users/nilaysinghsolanki/Downloads/S&P 500 Historical Data.csv")

    # Validate S&P 500 data
    if sp500_df.empty or sp500_df['Date'].isna().all():
        raise ValueError("S&P 500 data is empty or all dates are invalid")
    if 'Price' not in sp500_df.columns or 'Date' not in sp500_df.columns:
        raise ValueError("S&P 500 data must contain 'Date' and 'Price' columns")

    # Convert 'Price' to numeric, handling commas
    sp500_df['Price'] = pd.to_numeric(sp500_df['Price'].str.replace(',', ''), errors='coerce')

    # Drop rows with NaN values in 'Price' or 'Date'
    sp500_df = sp500_df.dropna(subset=['Price', 'Date'])

    # Log DataFrame shape for debugging
    st.write(f"S&P 500 DataFrame shape after loading and cleaning: {sp500_df.shape}")
    st.write(f"S&P 500 Date range: {sp500_df['Date'].min()} to {sp500_df['Date'].max()}")

    # Filter by date range
    sp500_df = sp500_df[(sp500_df['Date'] >= start_date) & (sp500_df['Date'] <= end_date)]

    # Validate S&P 500 data after filtering
    if sp500_df.empty:
        raise ValueError(f"S&P 500 data is empty after filtering by date range {start_date} to {end_date}")

    # Normalize S&P 500 data
    first_price = sp500_df['Price'].iloc[0]
    if first_price == 0:
        raise ValueError("First S&P 500 price is zero, cannot normalize")
    sp500_df['Normalized'] = sp500_df['Price'] / first_price

    return etf_df, sp500_df

# ---- TAB 1: Main Dashboard (Unchanged) ----
with tab1:
    st.title("ETF Portfolio Optimization Dashboard")

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_sectors = st.sidebar.multiselect("Select Sector(s):", sectors, default=sectors)

    # Date Range Filter
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input("Start Date", datetime(2014, 1, 1))
    end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

    # Portfolio Performance (LSTM Only)
    st.subheader("Portfolio Performance Metrics (Annualized)")
    portfolios = ['Model 1', 'Model 2', 'Model 3']
    perf_data = pd.DataFrame({
        'Portfolio': portfolios,
        'Annualized Return (%)': [4.72, 4.42, 3.77],
        'Annualized Volatility (%)': [3.83, 1.19, 1.43],
        'Sharpe Ratio': [0.31, 0.74, 0.16],
        'VaR (5%)': [-0.35, -0.11, -0.12],
        'Expected Shortfall (5%)': [-0.54, -0.14, -0.19],
        'Alpha (%)': [0.40, 0.70, 0.01]
    })
    st.dataframe(perf_data.style.set_properties(**{'border': '1px solid black'}).format(precision=3), use_container_width=True)

    fig_bar = px.bar(
        perf_data,
        x='Portfolio',
        y=['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'VaR (5%)', 'Expected Shortfall (5%)', 'Alpha (%)'],
        title="Portfolio Performance Comparison (Annualized Metrics)",
        barmode='group',
        height=500,
        color_discrete_map={
            'Annualized Return (%)': '#1f77b4',
            'Annualized Volatility (%)': '#ff7f0e',
            'Sharpe Ratio': '#2ca02c',
            'VaR (5%)': '#d62728',
            'Expected Shortfall (5%)': '#9467bd',
            'Alpha (%)': '#8c564b'
        }
    )
    fig_bar.update_layout(
        xaxis_title="Portfolio",
        yaxis_title="Value",
        legend_title="Metric",
        bargap=0.2
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Comparison of Methods Across Models (Bar Graph)
    st.subheader("Comparison of Methods Across Models")
    method_comparison_data = pd.DataFrame({
        'Model': ['Model 1', 'Model 1', 'Model 1', 'Model 2', 'Model 2', 'Model 2', 'Model 3', 'Model 3', 'Model 3'],
        'Method': ['LSTM', 'Transformer', 'RNN'] * 3,
        'Annualized Return (%)': [4.72, 12.48, 12.09, 4.42, 4.65, 4.04, 3.77, 5.40, 14.43],
        'Annualized Volatility (%)': [3.83, 9.65, 20.69, 1.19, 1.61, 1.40, 1.43, 2.06, 20.39],
        'Sharpe Ratio': [0.31, 0.93, 0.41, 0.74, 0.69, 0.36, 0.16, 0.90, 0.53],
        'VaR (5%)': [-0.35, -0.97, -1.90, -0.11, -0.16, -0.12, -0.12, -0.16, -1.99],
        'Expected Shortfall (5%)': [-0.54, -1.39, -3.00, -0.14, -0.21, -0.18, -0.19, -0.29, -3.00],
        'Alpha (%)': [0.40, 7.27, 3.20, 0.70, 0.86, 0.23, 0.01, 1.36, 6.32]
    })

    # Dropdown to select metric for comparison
    metric_options = ['Annualized Return (%)', 'Annualized Volatility (%)', 'Sharpe Ratio', 'VaR (5%)', 'Expected Shortfall (5%)', 'Alpha (%)']
    selected_metric = st.selectbox("Select Metric to Compare Across Methods:", metric_options)

    # Plot the comparison of methods (Bar Graph)
    fig_method_comparison = px.bar(
        method_comparison_data,
        x='Model',
        y=selected_metric,
        color='Method',
        title=f"Comparison of {selected_metric} Across Methods",
        barmode='group',
        height=500,
        color_discrete_map={
            'LSTM': '#1f77b4',
            'Transformer': '#ff7f0e',
            'RNN': '#2ca02c'
        }
    )
    fig_method_comparison.update_layout(
        xaxis_title="Model",
        yaxis_title=selected_metric,
        legend_title="Method",
        bargap=0.2,
        font=dict(size=14),
        title_font=dict(size=16),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_method_comparison, use_container_width=True)

    # Bubble Chart: Alpha vs Volatility
    st.subheader("Bubble Chart: Alpha vs Volatility (Bubble Size = Return)")
    method_comparison_data['Label'] = method_comparison_data['Model'].str.replace('Model ', 'Mod')

    # Create the bubble chart
    fig_bubble = go.Figure()

    # Add scatter traces for each method with different shapes
    for method, marker in zip(['LSTM', 'Transformer', 'RNN'], ['circle', 'square', 'triangle-up']):
        df = method_comparison_data[method_comparison_data['Method'] == method]
        fig_bubble.add_trace(
            go.Scatter(
                x=df['Annualized Volatility (%)'],
                y=df['Alpha (%)'],
                mode='markers+text',
                name=method,
                marker=dict(
                    size=df['Annualized Return (%)'] * 3,
                    symbol=marker,
                    color={
                        'LSTM': '#1f77b4',
                        'Transformer': '#2ca02c',
                        'RNN': '#ff7f0e'
                    }[method],
                    line=dict(width=1, color='black')
                ),
                text=df['Label'],
                textposition='middle center',
                textfont=dict(size=12, color='white')
            )
        )

    # Update layout to match the provided chart
    fig_bubble.update_layout(
        title="Bubble Chart: Alpha vs Volatility (Bubble Size = Return)",
        xaxis_title="Volatility (%)",
        yaxis_title="Alpha (%)",
        xaxis=dict(range=[0, 21], gridcolor='black', zeroline=False, tickvals=[0, 2.6, 5, 7.6, 10, 12.6, 15, 17.6, 20]),
        yaxis=dict(range=[0, 8], gridcolor='black', zeroline=False, tickvals=[0, 1, 2, 3, 4, 5, 6, 7, 8]),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black', size=14),
        title_font=dict(size=16, color='white'),
        legend=dict(
            title="Model",
            font=dict(color='white'),
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0
        ),
        showlegend=True,
        height=500
    )

    st.plotly_chart(fig_bubble, use_container_width=True)

    # Sector Weight Distribution
    st.subheader("Aggregating Portfolio Weights by Sector")
    weight_data = pd.DataFrame({
        'Model': ['Model 1', 'Model 1', 'Model 1', 'Model 1', 'Model 1',
                  'Model 2', 'Model 2', 'Model 2', 'Model 2', 'Model 2',
                  'Model 3', 'Model 3', 'Model 3', 'Model 3', 'Model 3'],
        'Sector': ['Energy', 'Financials', 'Real Estate', 'Healthcare', 'Technology'] * 3,
        'Weight (%)': [
            88.67, 0.00, 0.00, 10.32, 1.00,
            20.40, 71.31, 5.21, 1.81, 1.27,
            56.51, 30.13, 8.69, 2.26, 2.41
        ]
    })
    fig_weights = px.bar(
        weight_data,
        x='Model',
        y='Weight (%)',
        color='Sector',
        title="Sector Weight Distribution Across Models",
        height=500,
        color_discrete_map={
            'Energy': '#ff7f0e',
            'Financials': '#9467bd',
            'Real Estate': '#1f77b4',
            'Healthcare': '#d62728',
            'Technology': '#2ca02c'
        }
    )
    fig_weights.update_layout(
        barmode='stack',
        xaxis_title="Model",
        yaxis_title="Weight (%)",
        legend_title="Sector",
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_weights, use_container_width=True)

    # Time Series Plot
    st.subheader("Normalized Sector-Wise ETF vs S&P 500 Price Movement (2014–2024)")

    # Load data
    try:
        etf_df, sp500_df = load_data(start_date, end_date)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

    # Map sectors to tickers
    sector_to_ticker = {
        'Energy': 'XLE',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Real Estate': 'VNQ',
        'Technology': 'XLK'
    }
    # Adjust Healthcare ticker to ROBO since that's what's in the data
    sector_to_ticker['Healthcare'] = 'ROBO'

    # Define colors and labels to match the desired plot
    sector_styles = {
        'Energy': {'color': 'orange', 'label': 'Energy ETF'},
        'Financials': {'color': 'purple', 'label': 'Financials ETF'},
        'Healthcare': {'color': 'pink', 'label': 'Healthcare ETF'},
        'Real Estate': {'color': 'blue', 'label': 'Real Estate ETF'},
        'Technology': {'color': 'black', 'label': 'Technology ETF'}
    }

    selected_tickers = [sector_to_ticker[sector] for sector in selected_sectors if sector in sector_to_ticker]

    # Validate tickers
    available_tickers = etf_df['Ticker'].unique()
    selected_tickers = [ticker for ticker in selected_tickers if ticker in available_tickers]
    missing_tickers = [ticker for ticker in sector_to_ticker.values() if ticker not in available_tickers]
    if missing_tickers:
        st.warning(f"Missing data for the following tickers: {', '.join(missing_tickers)}. "
                   "To include these sectors (e.g., Real Estate - VNQ), please append their data to 'all_etf_data (1).csv' in the format: "
                   "Ticker,Sector,Date,Open,High,Low,Close,Volume. For example: "
                   "VNQ,Real Estate,01-01-2014,60.50,61.20,60.10,60.85,1234567")
    if not selected_tickers:
        st.warning("No valid tickers selected or available for the chosen sectors.")
        st.stop()

    # Create Plotly figure
    fig = go.Figure()

    # Plot selected ETFs
    plotted_sectors = []
    for sector in selected_sectors:
        ticker = sector_to_ticker.get(sector)
        if ticker in selected_tickers:
            ticker_data = etf_df[etf_df['Ticker'] == ticker]
            if not ticker_data.empty:
                style = sector_styles.get(sector, {'color': 'grey', 'label': sector})
                fig.add_trace(
                    go.Scatter(
                        x=ticker_data['Date'],
                        y=ticker_data['Normalized'],
                        mode='lines',
                        name=style['label'],
                        line=dict(color=style['color'])
                    )
                )
                plotted_sectors.append(sector)

    # Plot S&P 500
    if not sp500_df.empty:
        fig.add_trace(
            go.Scatter(
                x=sp500_df['Date'],
                y=sp500_df['Normalized'],
                mode='lines',
                name='S&P 500 Index',
                line=dict(color='cyan', dash='dash')
            )
        )
    else:
        st.warning("S&P 500 data is unavailable for the selected date range. It may only be available up to 2024-12-31.")

    # Update plot layout
    fig.update_layout(
        title="Normalized Sector-Wise ETF vs S&P 500 Price Movement (2014–2024)",
        xaxis_title="Date",
        yaxis_title="Normalized Price (Base = 1)",
        legend_title="Legend",
        hovermode="x unified",
        yaxis=dict(range=[0, 5])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Heatmaps
    st.subheader("Heatmaps: Sentiment, Macro, Returns")
    n = len(plotted_sectors) if plotted_sectors else 1
    sentiment_data = pd.DataFrame(np.random.rand(n, n), index=plotted_sectors or ['No Data'], columns=[f"Sent_{i}" for i in range(1, n+1)])
    macro_data = pd.DataFrame(np.random.rand(n, n), index=plotted_sectors or ['No Data'], columns=[f"Macro_{i}" for i in range(1, n+1)])
    returns_data = pd.DataFrame(np.random.rand(n, n), index=plotted_sectors or ['No Data'], columns=[f"Return_{i}" for i in range(1, n+1)])

    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.imshow(sentiment_data, color_continuous_scale='RdBu', title="Sentiment Sensitivity")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.imshow(macro_data, color_continuous_scale='Viridis', title="Macroeconomic Sensitivity")
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        fig3 = px.imshow(returns_data, color_continuous_scale='Plasma', title="Return Sensitivity")
        st.plotly_chart(fig3, use_container_width=True)

# ---- TAB 2: Data Pipeline & Sentiment Analysis ----
with tab2:
    st.title("Data Pipeline and Sentiment Analysis")
    st.subheader("\U0001F4C8 Data Foundation Process Flow")

    pipeline_img = Image.open("/Users/nilaysinghsolanki/Downloads/DSPipiline.png")
    st.image(pipeline_img, caption="Data Foundation Process Flow", output_format="PNG")

    st.subheader("Distribution of Unique News Articles Across Different Sectors")
    sector_data = pd.DataFrame({
        "Sector": sectors,
        "Percentage": [16.78, 22.12, 24.08, 7.31, 29.72]  # Updated percentages in the same order as sectors list
    })
    fig_donut = px.pie(sector_data, values='Percentage', names='Sector',
                       hole=0.5,
                       title="Cleaned News Distribution by Sector",
                       color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("Sentiment Distribution")

    st.markdown("### Sentiment Distribution per Sector")
    selected_sentiment_sector = st.selectbox("Select Sector for Sentiment Distribution:", sectors)
    
    # Sentiment distribution data from the provided table
    sentiment_distribution = pd.DataFrame({
        'Sector': ['Energy', 'Financials', 'Healthcare', 'Real Estate', 'Technology'],
        'Positive (%)': [24.48, 21.79, 30.33, 21.90, 24.33],
        'Negative (%)': [19.24, 18.18, 15.82, 16.18, 16.22],
        'Neutral (%)': [56.28, 60.03, 53.85, 61.92, 59.45]
    })

    # Filter for the selected sector and melt the data for plotting
    sector_dist = sentiment_distribution[sentiment_distribution['Sector'] == selected_sentiment_sector].melt(
        id_vars='Sector', 
        value_vars=['Positive (%)', 'Negative (%)', 'Neutral (%)'],
        var_name='Sentiment', 
        value_name='Percentage'
    )
    
    # Create the bar chart
    fig_dist_sector = px.bar(
        sector_dist, 
        x='Sentiment', 
        y='Percentage', 
        color='Sentiment',
        title=f"Sentiment Distribution - {selected_sentiment_sector}",
        color_discrete_map={'Positive (%)': 'green', 'Negative (%)': 'red', 'Neutral (%)': 'gray'},
        height=400
    )
    fig_dist_sector.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Percentage (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=True
    )
    st.plotly_chart(fig_dist_sector, use_container_width=True)