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
    
    # Multi-stock comparison section
    st.divider()
    st.subheader("Compare Multiple Stocks")
    comparison_symbols = st.text_input(
        "Enter stock symbols (comma-separated)",
        value="AAPL,MSFT,GOOGL",
        help="Enter 2-5 stock symbols separated by commas"
    )
    compare_button = st.button("Compare Stocks", type="secondary")
    
    # Stock screener section
    st.divider()
    st.subheader("Stock Screener")
    screener_symbols = st.text_area(
        "Enter stock symbols to screen (comma-separated)",
        value="AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,AMD,NFLX,DIS",
        help="Enter stock symbols separated by commas"
    )
    
    with st.expander("Screening Criteria"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_pe = st.number_input("Min P/E Ratio", value=0.0, step=1.0, help="Minimum P/E ratio")
            max_pe = st.number_input("Max P/E Ratio", value=100.0, step=1.0, help="Maximum P/E ratio")
            
            min_market_cap = st.number_input("Min Market Cap (Billions)", value=0.0, step=10.0, help="Minimum market cap in billions")
            max_market_cap = st.number_input("Max Market Cap (Billions)", value=10000.0, step=100.0, help="Maximum market cap in billions")
        
        with col2:
            min_volume = st.number_input("Min Avg Volume (Millions)", value=0.0, step=1.0, help="Minimum average volume in millions")
            
            min_dividend = st.number_input("Min Dividend Yield (%)", value=0.0, step=0.5, help="Minimum dividend yield percentage")
    
    screen_button = st.button("Screen Stocks", type="secondary")

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

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return macd, signal_line, histogram

def create_price_chart(hist_data, symbol, time_range, show_sma20=False, show_sma50=False, show_sma200=False):
    """Create interactive price chart with optional technical indicators"""
    fig = go.Figure()
    
    # Main price line
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=hist_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add SMA indicators if requested
    if show_sma20:
        sma20 = calculate_sma(hist_data, 20)
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=sma20,
            mode='lines',
            name='SMA 20',
            line=dict(color='#ff7f0e', width=1, dash='dash')
        ))
    
    if show_sma50:
        sma50 = calculate_sma(hist_data, 50)
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=sma50,
            mode='lines',
            name='SMA 50',
            line=dict(color='#2ca02c', width=1, dash='dash')
        ))
    
    if show_sma200:
        sma200 = calculate_sma(hist_data, 200)
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=sma200,
            mode='lines',
            name='SMA 200',
            line=dict(color='#d62728', width=1, dash='dash')
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

def create_rsi_chart(hist_data, symbol, time_range):
    """Create RSI chart"""
    rsi = calculate_rsi(hist_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=rsi,
        mode='lines',
        name='RSI',
        line=dict(color='#9467bd', width=2)
    ))
    
    # Add overbought/oversold reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    fig.update_layout(
        title=f"{symbol} RSI (Relative Strength Index) - {time_range}",
        xaxis_title="Date",
        yaxis_title="RSI",
        template='plotly_white',
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_macd_chart(hist_data, symbol, time_range):
    """Create MACD chart"""
    macd, signal_line, histogram = calculate_macd(hist_data)
    
    fig = go.Figure()
    
    # MACD line
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=macd,
        mode='lines',
        name='MACD',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Signal line
    fig.add_trace(go.Scatter(
        x=hist_data.index,
        y=signal_line,
        mode='lines',
        name='Signal',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    # Histogram
    colors = ['red' if val < 0 else 'green' for val in histogram]
    fig.add_trace(go.Bar(
        x=hist_data.index,
        y=histogram,
        name='Histogram',
        marker_color=colors,
        opacity=0.5
    ))
    
    fig.update_layout(
        title=f"{symbol} MACD - {time_range}",
        xaxis_title="Date",
        yaxis_title="MACD",
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

def create_comparison_chart(symbols_data, time_range):
    """Create overlay price comparison chart for multiple stocks"""
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (symbol, data) in enumerate(symbols_data.items()):
        hist_data = data['historical']
        # Normalize to percentage change from first day
        if len(hist_data) > 0:
            first_price = hist_data['Close'].iloc[0]
            normalized_prices = ((hist_data['Close'] / first_price) - 1) * 100
            
            fig.add_trace(go.Scatter(
                x=hist_data.index,
                y=normalized_prices,
                mode='lines',
                name=symbol,
                line=dict(color=colors[idx % len(colors)], width=2)
            ))
    
    fig.update_layout(
        title=f"Stock Price Comparison (% Change) - {time_range}",
        xaxis_title="Date",
        yaxis_title="% Change from Start",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def get_comparison_metrics(symbols):
    """Get key metrics for multiple stocks"""
    comparison_data = {}
    failed_symbols = []
    
    for symbol in symbols:
        stock, error = get_stock_info(symbol.strip().upper())
        if not error and stock:
            info = stock.info
            comparison_data[symbol.upper()] = {
                'Company': info.get('longName', symbol),
                'Current Price': f"${info.get('regularMarketPrice', 0):.2f}",
                'Market Cap': f"${info.get('marketCap', 0):,}" if info.get('marketCap') else 'N/A',
                'P/E Ratio': f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A',
                'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
                '52W High': f"${info.get('fiftyTwoWeekHigh', 0):.2f}" if info.get('fiftyTwoWeekHigh') else 'N/A',
                '52W Low': f"${info.get('fiftyTwoWeekLow', 0):.2f}" if info.get('fiftyTwoWeekLow') else 'N/A',
                'Beta': f"{info.get('beta', 0):.2f}" if info.get('beta') else 'N/A'
            }
        else:
            failed_symbols.append(symbol.upper())
    
    return comparison_data, failed_symbols

def get_fundamental_data(stock):
    """Fetch fundamental financial statements"""
    try:
        # Get financial statements
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        return {
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow
        }
    except Exception as e:
        return None

def format_financial_data(df, num_columns=4):
    """Format financial data for display"""
    if df is None or df.empty:
        return None
    
    # Transpose to have dates as rows
    df_display = df.T
    
    # Take only the most recent periods
    df_display = df_display.head(num_columns)
    
    # Format large numbers to millions/billions
    df_formatted = df_display.copy()
    for col in df_formatted.columns:
        if df_formatted[col].dtype in ['float64', 'int64']:
            df_formatted[col] = df_formatted[col].apply(
                lambda x: f"${x/1e9:.2f}B" if abs(x) >= 1e9 else f"${x/1e6:.2f}M" if abs(x) >= 1e6 else f"${x:,.0f}" if pd.notna(x) else "N/A"
            )
    
    return df_formatted

def screen_stocks(symbols, criteria):
    """Screen stocks based on given criteria"""
    screened_stocks = []
    failed_symbols = []
    
    for symbol in symbols:
        stock, error = get_stock_info(symbol.strip().upper())
        if not error and stock:
            info = stock.info
            
            # Extract metrics
            pe_ratio = info.get('trailingPE')
            market_cap = info.get('marketCap')
            avg_volume = info.get('averageVolume')
            dividend_yield = info.get('dividendYield', 0)
            
            # Apply filters
            passes = True
            
            # P/E Ratio filter
            if pe_ratio is not None:
                if pe_ratio < criteria['min_pe'] or pe_ratio > criteria['max_pe']:
                    passes = False
            elif criteria['min_pe'] > 0:  # If min_pe is set but PE is None, exclude
                passes = False
            
            # Market Cap filter (convert to billions)
            if market_cap is not None:
                market_cap_billions = market_cap / 1e9
                if market_cap_billions < criteria['min_market_cap'] or market_cap_billions > criteria['max_market_cap']:
                    passes = False
            else:
                passes = False
            
            # Average Volume filter (convert to millions)
            if avg_volume is not None:
                avg_volume_millions = avg_volume / 1e6
                if avg_volume_millions < criteria['min_volume']:
                    passes = False
            else:
                passes = False
            
            # Dividend Yield filter
            if dividend_yield is not None:
                dividend_yield_pct = dividend_yield * 100
                if dividend_yield_pct < criteria['min_dividend']:
                    passes = False
            elif criteria['min_dividend'] > 0:
                passes = False
            
            if passes:
                screened_stocks.append({
                    'Symbol': symbol.upper(),
                    'Company': info.get('longName', symbol),
                    'Price': f"${info.get('regularMarketPrice', 0):.2f}",
                    'P/E': f"{pe_ratio:.2f}" if pe_ratio else 'N/A',
                    'Market Cap': f"${market_cap/1e9:.2f}B" if market_cap else 'N/A',
                    'Avg Volume': f"{avg_volume/1e6:.2f}M" if avg_volume else 'N/A',
                    'Dividend Yield': f"{dividend_yield*100:.2f}%" if dividend_yield else '0.00%'
                })
        else:
            failed_symbols.append(symbol.upper())
    
    return screened_stocks, failed_symbols

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
                    # Technical Indicators Controls
                    st.subheader("📊 Technical Indicators")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        show_sma20 = st.checkbox("SMA 20", value=False, help="20-day Simple Moving Average")
                    with col2:
                        show_sma50 = st.checkbox("SMA 50", value=False, help="50-day Simple Moving Average")
                    with col3:
                        show_sma200 = st.checkbox("SMA 200", value=False, help="200-day Simple Moving Average")
                    
                    # Create tabs for different charts
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Chart", "RSI", "MACD", "Volume", "Data Table"])
                    
                    with tab1:
                        price_chart = create_price_chart(
                            hist_data, 
                            stock_symbol, 
                            selected_range,
                            show_sma20=show_sma20,
                            show_sma50=show_sma50,
                            show_sma200=show_sma200
                        )
                        st.plotly_chart(price_chart, use_container_width=True)
                    
                    with tab2:
                        rsi_chart = create_rsi_chart(hist_data, stock_symbol, selected_range)
                        st.plotly_chart(rsi_chart, use_container_width=True)
                        st.info("📌 RSI > 70 indicates overbought conditions, RSI < 30 indicates oversold conditions")
                    
                    with tab3:
                        macd_chart = create_macd_chart(hist_data, stock_symbol, selected_range)
                        st.plotly_chart(macd_chart, use_container_width=True)
                        st.info("📌 MACD crossovers signal potential buy/sell opportunities")
                    
                    with tab4:
                        volume_chart = create_volume_chart(hist_data, stock_symbol, selected_range)
                        st.plotly_chart(volume_chart, use_container_width=True)
                    
                    with tab5:
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
                
                # Fundamental Analysis Section
                st.header("📊 Fundamental Analysis")
                
                with st.spinner("Fetching fundamental data..."):
                    fundamental_data = get_fundamental_data(stock)
                    
                    if fundamental_data:
                        fund_tab1, fund_tab2, fund_tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
                        
                        with fund_tab1:
                            st.subheader("Income Statement")
                            income_df = format_financial_data(fundamental_data['income_statement'])
                            if income_df is not None and not income_df.empty:
                                st.dataframe(income_df, width='stretch')
                            else:
                                st.info("Income statement data not available for this stock")
                        
                        with fund_tab2:
                            st.subheader("Balance Sheet")
                            balance_df = format_financial_data(fundamental_data['balance_sheet'])
                            if balance_df is not None and not balance_df.empty:
                                st.dataframe(balance_df, width='stretch')
                            else:
                                st.info("Balance sheet data not available for this stock")
                        
                        with fund_tab3:
                            st.subheader("Cash Flow Statement")
                            cashflow_df = format_financial_data(fundamental_data['cash_flow'])
                            if cashflow_df is not None and not cashflow_df.empty:
                                st.dataframe(cashflow_df, width='stretch')
                            else:
                                st.info("Cash flow data not available for this stock")
                    else:
                        st.info("Fundamental data not available for this stock")

# Multi-stock comparison section
if compare_button:
    if not comparison_symbols:
        st.warning("Please enter stock symbols to compare")
    else:
        # Parse and deduplicate symbols
        symbols_list = [s.strip().upper() for s in comparison_symbols.split(',') if s.strip()]
        original_count = len(symbols_list)
        symbols_list = list(dict.fromkeys(symbols_list))  # Remove duplicates while preserving order
        
        # Notify user if duplicates were removed
        if len(symbols_list) < original_count:
            duplicates_removed = original_count - len(symbols_list)
            st.info(f"ℹ️ Removed {duplicates_removed} duplicate symbol(s). Comparing {len(symbols_list)} unique stocks.")
        
        if len(symbols_list) < 2:
            st.warning("Please enter at least 2 unique stock symbols to compare")
        elif len(symbols_list) > 5:
            st.warning("Please enter no more than 5 stock symbols to compare")
        else:
            with st.spinner("Fetching comparison data..."):
                # Get comparison metrics
                comparison_data, failed_symbols = get_comparison_metrics(symbols_list)
                
                # Display warnings for failed symbols
                if failed_symbols:
                    st.warning(f"⚠️ Unable to fetch data for: {', '.join(failed_symbols)}")
                
                if len(comparison_data) == 0:
                    st.error("Unable to fetch data for any of the provided symbols. Please check your symbols and try again.")
                elif len(comparison_data) < 2:
                    st.error(f"Only {len(comparison_data)} valid symbol found. Need at least 2 stocks for comparison.")
                else:
                    # Update header with actual count
                    st.header(f"📊 Comparing {len(comparison_data)} Stocks: {', '.join(comparison_data.keys())}")
                    
                    # Display comparison table
                    st.subheader("📋 Side-by-Side Metrics Comparison")
                    comparison_df = pd.DataFrame(comparison_data).T
                    st.dataframe(comparison_df, width='stretch')
                    
                    # Get historical data for comparison chart
                    st.subheader("📈 Price Performance Comparison")
                    days = time_ranges[selected_range]
                    symbols_data = {}
                    
                    for symbol in comparison_data.keys():
                        stock, error = get_stock_info(symbol)
                        if not error:
                            hist_data, hist_error = get_historical_data(stock, days)
                            if not hist_error and hist_data is not None:
                                symbols_data[symbol] = {
                                    'historical': hist_data
                                }
                    
                    if len(symbols_data) > 0:
                        comparison_chart = create_comparison_chart(symbols_data, selected_range)
                        st.plotly_chart(comparison_chart, use_container_width=True)
                        
                        st.info("📌 Chart shows percentage change from the start of the selected time period, allowing easy comparison of relative performance")
                        
                        # Export comparison data
                        st.subheader("💾 Export Comparison Data")
                        comparison_csv = comparison_df.to_csv()
                        st.download_button(
                            label="📋 Download Comparison Data (CSV)",
                            data=comparison_csv,
                            file_name=f"stock_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Unable to fetch historical data for comparison")

# Stock screener section
if screen_button:
    if not screener_symbols:
        st.warning("Please enter stock symbols to screen")
    else:
        symbols_list = [s.strip().upper() for s in screener_symbols.replace('\n', ',').split(',') if s.strip()]
        
        if len(symbols_list) == 0:
            st.warning("Please enter at least one stock symbol to screen")
        else:
            st.header("🔍 Stock Screener Results")
            
            # Prepare criteria dictionary
            criteria = {
                'min_pe': min_pe,
                'max_pe': max_pe,
                'min_market_cap': min_market_cap,
                'max_market_cap': max_market_cap,
                'min_volume': min_volume,
                'min_dividend': min_dividend
            }
            
            with st.spinner(f"Screening {len(symbols_list)} stocks..."):
                screened_results, failed = screen_stocks(symbols_list, criteria)
                
                if failed:
                    st.warning(f"⚠️ Unable to fetch data for {len(failed)} symbol(s): {', '.join(failed[:5])}" + 
                              (f" and {len(failed)-5} more" if len(failed) > 5 else ""))
                
                if len(screened_results) == 0:
                    st.info(f"📊 No stocks matched the screening criteria out of {len(symbols_list) - len(failed)} stocks analyzed")
                    
                    # Show criteria summary
                    st.write("**Applied Criteria:**")
                    criteria_text = []
                    if min_pe > 0 or max_pe < 100:
                        criteria_text.append(f"• P/E Ratio: {min_pe} - {max_pe}")
                    if min_market_cap > 0 or max_market_cap < 10000:
                        criteria_text.append(f"• Market Cap: ${min_market_cap}B - ${max_market_cap}B")
                    if min_volume > 0:
                        criteria_text.append(f"• Avg Volume: ≥ {min_volume}M")
                    if min_dividend > 0:
                        criteria_text.append(f"• Dividend Yield: ≥ {min_dividend}%")
                    
                    for criterion in criteria_text:
                        st.write(criterion)
                else:
                    st.success(f"✅ Found {len(screened_results)} stock(s) matching your criteria out of {len(symbols_list) - len(failed)} analyzed")
                    
                    # Display results table
                    results_df = pd.DataFrame(screened_results)
                    st.dataframe(results_df, width='stretch', hide_index=True)
                    
                    # Export screener results
                    st.subheader("💾 Export Screener Results")
                    screener_csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Screener Results (CSV)",
                        data=screener_csv,
                        file_name=f"stock_screener_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )

# Footer information
st.markdown("---")
st.markdown("""
**Data Source:** Yahoo Finance via yfinance library  
**Note:** All financial data is for informational purposes only and should not be considered as investment advice.
""")
