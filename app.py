import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Stock Financial Data Analyzer",
    page_icon="📈",
    layout="wide"
)

# Main title
st.title("📈 Stock Financial Data Analyzer")
st.markdown("Get comprehensive financial analysis and interactive charts for any stock symbol")

# Sidebar for stock input and time range selection
with st.sidebar:
    st.header("Configuration")
    
    # Stock symbol input
    stock_symbol = st.text_input(
        "Enter Stock Symbol",
        value="AAPL",
        help="Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT)"
    ).upper()
    
    # Time range selection for charts
    time_ranges = {
        "1M": 30,
        "3M": 90,
        "6M": 180,
        "1Y": 365,
        "5Y": 1825
    }
    
    selected_range = st.selectbox(
        "Select Time Range for Charts",
        options=list(time_ranges.keys()),
        index=3  # Default to 1Y
    )
    
    analyze_button = st.button("Analyze Stock", type="primary")

def get_stock_info(symbol):
    """Fetch stock information from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Check if the stock exists by verifying we have basic info
        if not info or info.get('regularMarketPrice') is None:
            return None, "Invalid stock symbol or no data available"
            
        return stock, None
    except Exception as e:
        return None, f"Error fetching stock data: {str(e)}"

def get_financial_summary(stock_info):
    """Extract key financial metrics from stock info"""
    try:
        summary_data = {
            "Current Price": stock_info.get('regularMarketPrice', 'N/A'),
            "Previous Close": stock_info.get('previousClose', 'N/A'),
            "Market Cap": stock_info.get('marketCap', 'N/A'),
            "P/E Ratio": stock_info.get('trailingPE', 'N/A'),
            "Forward P/E": stock_info.get('forwardPE', 'N/A'),
            "Dividend Yield (%)": stock_info.get('dividendYield', 'N/A'),
            "52 Week High": stock_info.get('fiftyTwoWeekHigh', 'N/A'),
            "52 Week Low": stock_info.get('fiftyTwoWeekLow', 'N/A'),
            "Volume": stock_info.get('volume', 'N/A'),
            "Average Volume": stock_info.get('averageVolume', 'N/A'),
            "Beta": stock_info.get('beta', 'N/A'),
            "EPS": stock_info.get('trailingEps', 'N/A')
        }
        
        # Format certain values
        if summary_data["Market Cap"] != 'N/A':
            summary_data["Market Cap"] = f"${summary_data['Market Cap']:,}"
        
        if summary_data["Dividend Yield (%)"] != 'N/A':
            summary_data["Dividend Yield (%)"] = f"{summary_data['Dividend Yield (%)'] * 100:.2f}%"
        
        if summary_data["Current Price"] != 'N/A':
            summary_data["Current Price"] = f"${summary_data['Current Price']:.2f}"
            
        if summary_data["Previous Close"] != 'N/A':
            summary_data["Previous Close"] = f"${summary_data['Previous Close']:.2f}"
            
        if summary_data["52 Week High"] != 'N/A':
            summary_data["52 Week High"] = f"${summary_data['52 Week High']:.2f}"
            
        if summary_data["52 Week Low"] != 'N/A':
            summary_data["52 Week Low"] = f"${summary_data['52 Week Low']:.2f}"
        
        return summary_data
    except Exception as e:
        st.error(f"Error processing financial summary: {str(e)}")
        return None

def get_historical_data(stock, days):
    """Fetch historical stock data"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            return None, "No historical data available for this time range"
            
        return hist_data, None
    except Exception as e:
        return None, f"Error fetching historical data: {str(e)}"

def create_price_chart(hist_data, symbol, time_range):
    """Create interactive price chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title=f"{symbol} Stock Price - {time_range}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_volume_chart(hist_data, symbol, time_range):
    """Create interactive volume chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=hist_data['Volume'],
        name='Volume',
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title=f"{symbol} Trading Volume - {time_range}",
        xaxis_title="Date",
        yaxis_title="Volume",
        template='plotly_white',
        height=400
    )
    
    return fig

def prepare_csv_data(summary_data, hist_data, symbol):
    """Prepare data for CSV export"""
    # Create summary DataFrame
    summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
    summary_df.insert(0, 'Stock Symbol', symbol)
    
    # Prepare historical data
    hist_df = hist_data.reset_index()
    hist_df.insert(0, 'Stock Symbol', symbol)
    
    return summary_df, hist_df

# Main application logic
if analyze_button or stock_symbol:
    if not stock_symbol:
        st.warning("Please enter a stock symbol")
    else:
        with st.spinner(f"Fetching data for {stock_symbol}..."):
            stock, error = get_stock_info(stock_symbol)
            
            if error:
                st.error(error)
            else:
                # Get stock info and company name
                stock_info = stock.info
                company_name = stock_info.get('longName', stock_symbol)
                
                st.success(f"Successfully loaded data for {company_name} ({stock_symbol})")
                
                # Create two columns for layout
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.header("📊 Financial Summary")
                    summary_data = get_financial_summary(stock_info)
                    
                    if summary_data:
                        # Display summary as a formatted table
                        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
                        # Ensure all values are strings for proper display
                        summary_df['Value'] = summary_df['Value'].astype(str)
                        st.dataframe(summary_df, width='stretch', hide_index=True)
                
                with col2:
                    st.header("📈 Key Statistics")
                    # Additional metrics in a more visual format
                    current_price = stock_info.get('regularMarketPrice', 0)
                    prev_close = stock_info.get('previousClose', 0)
                    
                    if current_price and prev_close:
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100
                        
                        if change >= 0:
                            st.metric("Current Price", f"${current_price:.2f}", f"+${change:.2f} (+{change_percent:.2f}%)")
                        else:
                            st.metric("Current Price", f"${current_price:.2f}", f"${change:.2f} ({change_percent:.2f}%)")
                    
                    # Display additional metrics
                    market_cap = stock_info.get('marketCap')
                    if market_cap:
                        st.metric("Market Cap", f"${market_cap:,}")
                    
                    pe_ratio = stock_info.get('trailingPE')
                    if pe_ratio:
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}")
                
                # Fetch and display historical data
                st.header("📈 Historical Data & Charts")
                
                days = time_ranges[selected_range]
                hist_data, hist_error = get_historical_data(stock, days)
                
                if hist_error:
                    st.error(hist_error)
                else:
                    # Create tabs for different charts
                    tab1, tab2, tab3 = st.tabs(["Price Chart", "Volume Chart", "Data Table"])
                    
                    with tab1:
                        price_chart = create_price_chart(hist_data, stock_symbol, selected_range)
                        st.plotly_chart(price_chart, use_container_width=True)
                    
                    with tab2:
                        volume_chart = create_volume_chart(hist_data, stock_symbol, selected_range)
                        st.plotly_chart(volume_chart, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Historical Price Data")
                        # Format the historical data for display
                        display_data = hist_data.copy()
                        for col in ['Open', 'High', 'Low', 'Close']:
                            if col in display_data.columns:
                                display_data[col] = display_data[col].round(2)
                        
                        st.dataframe(display_data, width='stretch')
                    
                    # CSV Download Section
                    st.header("💾 Export Data")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Prepare CSV data
                        summary_df, hist_df = prepare_csv_data(summary_data, hist_data, stock_symbol)
                        
                        # Convert DataFrames to CSV
                        summary_csv = summary_df.to_csv(index=False)
                        historical_csv = hist_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📋 Download Summary Data (CSV)",
                            data=summary_csv,
                            file_name=f"{stock_symbol}_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    with col2:
                        st.download_button(
                            label="📈 Download Historical Data (CSV)",
                            data=historical_csv,
                            file_name=f"{stock_symbol}_historical_{selected_range}_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )

# Footer information
st.markdown("---")
st.markdown("""
**Data Source:** Yahoo Finance via yfinance library  
**Note:** All financial data is for informational purposes only and should not be considered as investment advice.
""")
