import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set page config
st.set_page_config(
    page_title="Intelligent Wealth Advisory Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WealthAdvisor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.rf_model = None
        self.lstm_model = None
        
    @st.cache_data
    def fetch_market_data(_self, symbol="^GSPC", period="10y"):
        """Fetch historical market data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None
    
    def create_features(self, data):
        """Create technical indicators and features"""
        df = data.copy()
        
        # Technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Price features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Range'] = df['High'] - df['Low']
        
        # Lag features
        for i in [1, 2, 3, 5, 10]:
            df[f'Return_Lag_{i}'] = df['Price_Change'].shift(i)
            
        return df.dropna()
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def prepare_ml_data(self, data, target_days=126):  # 6 months ≈ 126 trading days
        """Prepare data for machine learning models"""
        df = self.create_features(data)
        
        # Define features for Random Forest
        feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'Volatility', 'Volume_MA',
                       'High_Low_Pct', 'Price_Range'] + [f'Return_Lag_{i}' for i in [1, 2, 3, 5, 10]]
        
        # Create target (6-month forward returns)
        df['Target'] = df['Close'].pct_change(periods=target_days).shift(-target_days)
        
        # Clean data
        df = df.dropna()
        
        X = df[feature_cols]
        y = df['Target']
        
        return X, y, df
    
    def train_random_forest(self, X, y):
        """Train Random Forest model"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Predictions and metrics
        train_pred = self.rf_model.predict(X_train)
        test_pred = self.rf_model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'feature_importance': dict(zip(X.columns, self.rf_model.feature_importances_))
        }
    
    def prepare_lstm_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        prices = data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(prices)
        
        X, y = [], []
        for i in range(lookback, len(scaled_data) - 126):  # 126 days ahead prediction
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i+126, 0])  # 6 months ahead
            
        return np.array(X), np.array(y)
    
    def train_lstm(self, X, y):
        """Train LSTM model"""
        X = X.reshape((X.shape[0], X.shape[1], 1))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        history = self.lstm_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=50,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Predictions
        train_pred = self.lstm_model.predict(X_train)
        test_pred = self.lstm_model.predict(X_test)
        
        # Calculate R² scores
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'history': history.history
        }
    
    def monte_carlo_simulation(self, initial_investment, expected_return, volatility, 
                              years=10, simulations=1000):
        """Run Monte Carlo simulation for portfolio performance"""
        np.random.seed(42)
        
        # Parameters
        dt = 1/252  # Daily time step
        days = years * 252
        
        # Initialize results array
        results = np.zeros((simulations, days))
        results[:, 0] = initial_investment
        
        for i in range(1, days):
            # Generate random returns
            random_returns = np.random.normal(
                expected_return * dt,
                volatility * np.sqrt(dt),
                simulations
            )
            results[:, i] = results[:, i-1] * (1 + random_returns)
        
        return results
    
    def portfolio_optimization(self, risk_profile):
        """Generate portfolio allocation based on risk profile"""
        allocations = {
            'Conservative': {
                'Stocks': 30,
                'Bonds': 60,
                'Cash': 10,
                'expected_return': 0.06,
                'volatility': 0.08
            },
            'Moderate': {
                'Stocks': 60,
                'Bonds': 35,
                'Cash': 5,
                'expected_return': 0.08,
                'volatility': 0.12
            },
            'Aggressive': {
                'Stocks': 85,
                'Bonds': 10,
                'Cash': 5,
                'expected_return': 0.11,
                'volatility': 0.18
            }
        }
        return allocations.get(risk_profile, allocations['Moderate'])

def main():
    st.title("🏦 Intelligent Wealth Advisory Platform")
    st.markdown("Advanced portfolio analysis with AI-powered predictions and risk assessment")
    
    # Initialize advisor
    advisor = WealthAdvisor()
    
    # Sidebar
    st.sidebar.header("Portfolio Configuration")
    
    # Risk profile selection
    risk_profile = st.sidebar.selectbox(
        "Select Risk Profile",
        ['Conservative', 'Moderate', 'Aggressive']
    )
    
    # Investment amount
    investment_amount = st.sidebar.number_input(
        "Initial Investment ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=5000
    )
    
    # Time horizon
    time_horizon = st.sidebar.slider(
        "Investment Time Horizon (years)",
        min_value=1,
        max_value=30,
        value=10
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Market Analysis", 
        "🤖 AI Predictions", 
        "🎯 Monte Carlo Simulation", 
        "📈 Portfolio Optimization"
    ])
    
    # Tab 1: Market Analysis
    with tab1:
        st.header("S&P 500 Market Analysis")
        
        with st.spinner("Fetching market data..."):
            market_data = advisor.fetch_market_data()
        
        if market_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=market_data.index,
                    y=market_data['Close'],
                    mode='lines',
                    name='S&P 500',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title="S&P 500 Historical Performance",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Volume chart
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(
                    x=market_data.index,
                    y=market_data['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ))
                fig_vol.update_layout(
                    title="Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=400
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            # Market statistics
            st.subheader("Market Statistics")
            returns = market_data['Close'].pct_change().dropna()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${market_data['Close'][-1]:.2f}")
            with col2:
                st.metric("Annual Return", f"{returns.mean() * 252 * 100:.2f}%")
            with col3:
                st.metric("Volatility", f"{returns.std() * np.sqrt(252) * 100:.2f}%")
            with col4:
                st.metric("Sharpe Ratio", f"{(returns.mean() * 252) / (returns.std() * np.sqrt(252)):.2f}")
    
    # Tab 2: AI Predictions
    with tab2:
        st.header("AI-Powered Portfolio Predictions")
        
        if st.button("Train Models & Generate Predictions"):
            market_data = advisor.fetch_market_data()
            
            if market_data is not None:
                # Prepare data
                X, y, df = advisor.prepare_ml_data(market_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🌳 Random Forest Model")
                    with st.spinner("Training Random Forest..."):
                        rf_results = advisor.train_random_forest(X, y)
                    
                    st.write(f"**Test Accuracy (R²):** {rf_results['test_r2']:.3f}")
                    st.write(f"**Test RMSE:** {rf_results['test_rmse']:.4f}")
                    
                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': list(rf_results['feature_importance'].keys()),
                        'Importance': list(rf_results['feature_importance'].values())
                    }).sort_values('Importance', ascending=True)
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Feature Importance'
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                with col2:
                    st.subheader("🔗 LSTM Neural Network")
                    with st.spinner("Training LSTM..."):
                        X_lstm, y_lstm = advisor.prepare_lstm_data(market_data)
                        lstm_results = advisor.train_lstm(X_lstm, y_lstm)
                    
                    st.write(f"**Test Accuracy (R²):** {lstm_results['test_r2']:.3f}")
                    st.write(f"**Test RMSE:** {lstm_results['test_rmse']:.4f}")
                    
                    # Training history
                    if 'history' in lstm_results:
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=lstm_results['history']['loss'],
                            mode='lines',
                            name='Training Loss'
                        ))
                        fig_loss.add_trace(go.Scatter(
                            y=lstm_results['history']['val_loss'],
                            mode='lines',
                            name='Validation Loss'
                        ))
                        fig_loss.update_layout(
                            title='Model Training Loss',
                            xaxis_title='Epoch',
                            yaxis_title='Loss'
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                
                # Combined accuracy metric
                combined_accuracy = (rf_results['test_r2'] + lstm_results['test_r2']) / 2
                st.success(f"🎯 **Combined Model Accuracy: {combined_accuracy:.1%}**")
    
    # Tab 3: Monte Carlo Simulation
    with tab3:
        st.header("Monte Carlo Risk Analysis")
        
        # Get portfolio allocation
        portfolio = advisor.portfolio_optimization(risk_profile)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Portfolio Allocation")
            
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(portfolio.keys())[:3],  # Exclude expected_return and volatility
                values=[portfolio['Stocks'], portfolio['Bonds'], portfolio['Cash']],
                hole=.3
            )])
            fig_pie.update_layout(title=f"{risk_profile} Portfolio")
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.write(f"**Expected Return:** {portfolio['expected_return']:.1%}")
            st.write(f"**Volatility:** {portfolio['volatility']:.1%}")
        
        with col2:
            st.subheader("Simulation Results")
            
            if st.button("Run Monte Carlo Simulation (1,000 scenarios)"):
                with st.spinner("Running simulation..."):
                    results = advisor.monte_carlo_simulation(
                        investment_amount,
                        portfolio['expected_return'],
                        portfolio['volatility'],
                        time_horizon
                    )
                
                # Plot simulation results
                fig_mc = go.Figure()
                
                # Plot sample paths
                for i in range(0, len(results), 50):  # Plot every 50th simulation
                    fig_mc.add_trace(go.Scatter(
                        x=list(range(len(results[i]))),
                        y=results[i],
                        mode='lines',
                        line=dict(width=0.5, color='lightblue'),
                        showlegend=False
                    ))
                
                # Plot percentiles
                percentiles = np.percentile(results, [10, 50, 90], axis=0)
                
                fig_mc.add_trace(go.Scatter(
                    x=list(range(len(percentiles[1]))),
                    y=percentiles[1],
                    mode='lines',
                    line=dict(width=3, color='red'),
                    name='Median (50th percentile)'
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=list(range(len(percentiles[0]))),
                    y=percentiles[0],
                    mode='lines',
                    line=dict(width=2, color='orange', dash='dash'),
                    name='10th percentile'
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=list(range(len(percentiles[2]))),
                    y=percentiles[2],
                    mode='lines',
                    line=dict(width=2, color='green', dash='dash'),
                    name='90th percentile'
                ))
                
                fig_mc.update_layout(
                    title='Monte Carlo Portfolio Simulation',
                    xaxis_title='Days',
                    yaxis_title='Portfolio Value ($)',
                    height=500
                )
                st.plotly_chart(fig_mc, use_container_width=True)
                
                # Summary statistics
                final_values = results[:, -1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Median Final Value", f"${np.median(final_values):,.0f}")
                with col2:
                    st.metric("10th Percentile", f"${np.percentile(final_values, 10):,.0f}")
                with col3:
                    st.metric("90th Percentile", f"${np.percentile(final_values, 90):,.0f}")
                with col4:
                    loss_probability = np.mean(final_values < investment_amount) * 100
                    st.metric("Loss Probability", f"{loss_probability:.1f}%")
    
    # Tab 4: Portfolio Optimization
    with tab4:
        st.header("Portfolio Recommendations")
        
        portfolio = advisor.portfolio_optimization(risk_profile)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recommended Allocation")
            
            allocation_df = pd.DataFrame({
                'Asset Class': ['Stocks', 'Bonds', 'Cash'],
                'Allocation (%)': [portfolio['Stocks'], portfolio['Bonds'], portfolio['Cash']],
                'Amount ($)': [
                    investment_amount * portfolio['Stocks'] / 100,
                    investment_amount * portfolio['Bonds'] / 100,
                    investment_amount * portfolio['Cash'] / 100
                ]
            })
            
            st.dataframe(allocation_df, use_container_width=True)
            
            # Risk metrics
            st.subheader("Risk Metrics")
            st.write(f"**Risk Profile:** {risk_profile}")
            st.write(f"**Expected Annual Return:** {portfolio['expected_return']:.1%}")
            st.write(f"**Expected Volatility:** {portfolio['volatility']:.1%}")
            st.write(f"**Sharpe Ratio Estimate:** {portfolio['expected_return']/portfolio['volatility']:.2f}")
        
        with col2:
            st.subheader("Risk-Return Comparison")
            
            # Compare different risk profiles
            profiles = ['Conservative', 'Moderate', 'Aggressive']
            returns = []
            risks = []
            
            for profile in profiles:
                p = advisor.portfolio_optimization(profile)
                returns.append(p['expected_return'])
                risks.append(p['volatility'])
            
            fig_risk = go.Figure()
            
            for i, profile in enumerate(profiles):
                color = 'red' if profile == risk_profile else 'lightblue'
                size = 15 if profile == risk_profile else 10
                
                fig_risk.add_trace(go.Scatter(
                    x=[risks[i]],
                    y=[returns[i]],
                    mode='markers+text',
                    text=[profile],
                    textposition='top center',
                    marker=dict(size=size, color=color),
                    name=profile
                ))
            
            fig_risk.update_layout(
                title='Risk-Return Profile',
                xaxis_title='Volatility (Risk)',
                yaxis_title='Expected Return',
                height=400
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.subheader("Investment Strategy")
            if risk_profile == 'Conservative':
                st.info("🛡️ Focus on capital preservation with steady, low-risk returns.")
            elif risk_profile == 'Moderate':
                st.info("⚖️ Balanced approach seeking growth while managing risk.")
            else:
                st.info("🚀 Aggressive growth strategy with higher risk tolerance.")

if __name__ == "__main__":
    main()