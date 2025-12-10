# ==============================================================================
# 🚨 DO NOT CHANGE – FROZEN FOR FORWARD TEST FT_001 🚨
# Any changes must be done in experiments/v3_ideas/
# Statistics Baseline: Mean R=0.12, Win Rate=45%, Max DD=10%
# ==============================================================================
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# ENUMS FOR REGIMES
# ============================================================================
class HMMRegime(Enum):
    LOW_VOL = 0
    HIGH_VOL = 1
    UNKNOWN = -1

class VolRegime(Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"

class TrendRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    RANGE = "RANGE"

class MRRegime(Enum):
    MR_STRONG = "MR_STRONG"
    MR_WEAK = "MR_WEAK"

class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    NONE = 0

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    # === Universe Selection ===
    # 2024-12-09: OPTIMIZED CORE 13 (High Liquidity)
    # Removed Toxic Minor Crosses which destroyed Sharpe (2.29 -> 0.27)
    symbols: List[str] = field(default_factory=lambda: [
        # --- FOREX MAJORS ---
        "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X", 
        "USDCAD=X", "AUDUSD=X", "NZDUSD=X",
        
        # --- FOREX CROSSES (High Liquidity Only) ---
        "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X",
        "EURAUD=X", "EURCHF=X"
        
        # REMOVED TOXIC PAIRS (Wide Spreads / Low Alpha):
        # AUDNZD, AUDCAD, CADJPY, NZDJPY, GBPCHF, 
        # GBPAUD, GBPCAD, EURNZD
    ])
    
    # === Data Frequency ===
    timeframe: str = "1h"
    
    # === Indicator Parameters ===
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Legacy Params (Required by FeatureEngine)
    ma_period: int = 50
    std_period: int = 50
    mom_lookback: int = 10
    vol_period: int = 20
    
    # === Regime Parameters ===
    vol_low_pct: float = 25.0
    vol_high_pct: float = 75.0
    fast_ma_period: int = 20
    slow_ma_period: int = 50
    trend_slope_threshold: float = 0.001
    
    # === Ensemble Signal Parameters ===
    # Weights sum to ~ 3.0 to 5.0, but Signal logic caps leverage at 1.0.
    w_alpha: float = 3.0       # Primary Driver (ML) -> Validated by Ablation Test (ML Only > Full)
    w_stat_arb: float = 0.0    # Disabled -> Validation showed 0 trades contribution.
    w_mr: float = 0.0
    w_mom: float = 0.0
    w_vol: float = 0.0
    w_candle: float = 0.0
    w_trend: float = 0.0
    
    z_long_threshold: float = -1.5
    z_short_threshold: float = 1.5
    z_exit_threshold: float = 0.5
    ensemble_threshold: float = 0.1
    
    # === Machine Learning Config ===
    use_alpha_engine: bool = True
    alpha_lookback_period: int = 150
    # CHANGED: Shortened to 1 (Next Bar) for easier prediction
    alpha_target_lookahead: int = 1 
    alpha_return_threshold: float = 0.001
    alpha_train_split: float = 0.7
    alpha_model_params: Dict = field(default_factory=lambda: {
        'objective': 'multi:softprob',
        'num_class': 3,
        'max_depth': 3,          # Reduced from 6 to prevent overfitting
        'eta': 0.1,
        'gamma': 0.2,            # Increased regularization (0.1 -> 0.2)
        'subsample': 0.7,        # Increased noise
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42
    })
    # THRESHOLD OPTIMIZATION: Signal Force (0.505)
    ml_prob_threshold_long: float = 0.505  # Lowered to force activity
    ml_prob_threshold_short: float = 0.505 # Lowered to force activity
    ml_target_horizon: int = 1
    
    # === Rank Signal Config (Alpha V3) ===
    use_rank_logic: bool = True  # ENABLED: Force Signal via Ranking
    rank_top_n: int = 1          # Long Top 1, Short Bottom 1
    
    # ROLLING WINDOW CONFIG
    # yfinance 1h data limit is ~730 days.
    # We train on first 80%, Test/Trade on last 20%.
    # ROLLING WINDOW CONFIG
    # yfinance 1h data limit is ~730 days.
    # We train on first 80%, Test/Trade on last 20%.
    ml_train_split_pct: float = 0.80
    ml_lookback_days: int = 729 # Max fetch for yfinance
    
    # === WFO Parameters (Walk-Forward Optimization) ===
    # For Hourly data (approx 7 trading hours/day * 20 days = 140 bars/month)
    wfo_train_bars: int = 500  # Reduced from 1500 to ensure fitting in test set
    wfo_test_bars: int = 100   # Reduced from 140     # Approx 1 month OOS prediction
    mode: str = "BACKTEST"       # "BACKTEST" (WFO) or "LIVE" (Full Train)
    
    # === Stat Arb Config ===
    use_stat_arb: bool = False # DISABLED (Ablation Result: Dead Weight)
    ou_threshold: float = 2.0
    coint_p_value: float = 0.05
    stat_arb_lookback: int = 200
    stat_arb_z_trigger: float = 2.0
    stat_arb_zscore_exit: float = 0.5
    
    # === Risk Management ===
    initial_balance: float = 100000.0
    account_leverage: float = 30.0 # FTMO Swing
    # RISK MANAGEMENT (FINAL SCIENTIFIC CALIBRATION)
    # RISK MANAGEMENT (FINAL SCIENTIFIC CALIBRATION)
    # RISK MANAGEMENT (FINAL SCIENTIFIC CALIBRATION)
    # Scaled 2024-12-09: 3.3% Risk (Monte Carlo Validated)
    # 95% CI Max Drawdown < 8.0%. Exp Return ~17%.
    risk_per_trade: float = 0.05         # 5.0% per trade (Final Live Setting)
    max_concurrent_trades: int = 10      # Focused (Reduced from 15)
    max_exposure_per_currency: int = 10 
    
    # === COSTS ===
    transaction_cost: float = 0.0005     # 5 basis points (0.05%) per side.
    
    # === Institutional Entry/Exit (No Retail Astrology) ===
    # 1. High Confidence Entry
    min_prob_margin: float = 0.05        # Lowered to 0.05 (Top ~7%)
    high_conf_threshold: float = 0.10    # Override regime blocks if confidence > this
    
    # 2. Signal-Driven Exits (Medallion Style)
    use_signal_decay_exit: bool = True   # ENABLED (Vital for Sharpe)
    signal_decay_threshold: float = 0.45 # Exit if signal drops below this
    emergency_sl_mult: float = 5.0       # Catastrophe protection only
    max_bars_in_trade: int = 30          # Time exit
    
    # Kelly Criterion
    use_kelly: bool = True
    kelly_fraction: float = 1.0
    min_win_rate_for_kelly: float = 0.51
    kelly_lookback: int = 50
    
    # Dynamic SL/TP (Removed - Legacy)
    use_dynamic_sltp: bool = False
    regime_sl_bonus: float = 0.0
    
    max_positions_per_symbol: int = 1
    
    # === FTMO Rules ===
    daily_loss_limit_pct: float = 5.0
    overall_loss_limit_pct: float = 10.0
    
    # === Advanced Risk ===
    drawdown_penalty_days: int = 5
    risk_penalty_factor: float = 0.5
    vol_scaling_factor: float = 0.5
    
    # === Monte Carlo ===
    mc_simulations: int = 1000
    mc_target_return: float = 10.0
    
    # === HMM ===
    use_hmm: bool = True
    hmm_n_components: int = 3
    hmm_min_samples: int = 100
    hmm_n_iter: int = 100
    hmm_covariance_type: str = 'diag'
    seed: int = 42

# ============================================================================
# DATA HANDLING
# ============================================================================
class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def download_single_symbol(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        try:
            print(f"  Downloading {symbol}...")
            df = yf.download(symbol, start=start, end=end, interval=self.config.timeframe, progress=False)
            if df.empty or len(df) < 100:
                print(f"    WARNING: {symbol} has insufficient data.")
                return None
            
            # Flatten columns if MultiIndex (Open, Close, etc.)
            if isinstance(df.columns, pd.MultiIndex):
                # If only one level has relevant names, extract
                # Usually Level 0 is Price Type, Level 1 is Ticker or vice versa
                # For single ticker download:
                if 'Close' in df.columns.get_level_values(0):
                     df.columns = df.columns.get_level_values(0)
                else: 
                     # Only keep the ticker level?
                     # yfinance structure varies. Simplest:
                     df = df.xs(symbol, axis=1, level=1, drop_level=True)
            
            # Basic cleaning
            df = df.dropna()
            print(f"    OK ({len(df)} bars)")
            return df
        except Exception as e:
            print(f"    ERROR: {e}")
            return None

    def load_data(self, start: str, end: str) -> Dict[str, pd.DataFrame]:
        print(f"\n[DOWNLOADING DATA]")
        print("-" * 40)
        data = {}
        for sym in self.config.symbols:
            df = self.download_single_symbol(sym, start, end)
            if df is not None:
                data[sym] = df
        
        print(f"\nSuccessfully downloaded: {len(data)}/{len(self.config.symbols)} symbols")
        
        # Align data
        if not data:
            raise ValueError("No data downloaded!")
            
        print(f"\n[ALIGNING DATA]")
        print("-" * 40)
        # Find common index interaction
        common_index = None
        for sym, df in data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        aligned_data = {}
        for sym, df in data.items():
            df_aligned = df.loc[common_index].copy()
            aligned_data[sym] = df_aligned
            print(f"  {sym}: {len(df_aligned)} bars")
        
        if len(common_index) == 0:
            raise ValueError("No overlapping data found!")
            
        print(f"\nCommon date range: {common_index.min()} to {common_index.max()}")
        print(f"Total aligned bars: {len(common_index)}")
        
        return aligned_data

# ============================================================================
# FEATURE ENGINEERING (PHASE 1)
# ============================================================================
class FeatureEngine:
    def __init__(self, config: Config):
        self.config = config
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # --- Basic Features ---
        df['MA'] = df['Close'].rolling(self.config.ma_period).mean()
        df['STD'] = df['Close'].rolling(self.config.std_period).std()
        
        # Z-Score
        df['Z_Score'] = (df['Close'] - df['MA']) / df['STD'].replace(0, np.nan)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(self.config.atr_period).mean()
        
        # Momentum
        df['Momentum'] = df['Close'].diff(self.config.mom_lookback)
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(self.config.vol_period).std()
        
        # --- New: Volatility Intensity (z-score style) ---
        vol = df['Volatility']
        vol_mean = vol.rolling(200).mean()
        vol_std  = vol.rolling(200).std()
        
        # Avoid division by zero
        df['Vol_Intensity'] = (vol - vol_mean) / vol_std.replace(0, np.nan)
        df['Vol_Intensity'] = df['Vol_Intensity'].fillna(0.0)
        
        # Percentile rank (0..1) over rolling window
        vol_rolling = vol.rolling(200)
        df['Vol_Pct'] = vol_rolling.rank(pct=True).fillna(0.5)
        
        # Update Vol_Regime buckets based on Percentile (Backward Compatibility)
        df['Vol_Regime'] = 'NORMAL'
        df['Vol_Regime_Num'] = 1
        
        df.loc[df['Vol_Pct'] <= 0.25, 'Vol_Regime'] = 'LOW'
        df.loc[df['Vol_Pct'] <= 0.25, 'Vol_Regime_Num'] = 0
        
        df.loc[df['Vol_Pct'] >= 0.75, 'Vol_Regime'] = 'HIGH'
        df.loc[df['Vol_Pct'] >= 0.75, 'Vol_Regime_Num'] = 2

        # Efficiency Ratio (Kaufman)
        change = df['Close'].diff(10).abs()
        volatility_sum = df['Close'].diff().abs().rolling(10).sum()
        df['Efficiency_Ratio'] = change / volatility_sum.replace(0, np.nan)
        
        # Regime Features
        df['Fast_MA'] = df['Close'].rolling(self.config.fast_ma_period).mean()
        df['Slow_MA'] = df['Close'].rolling(self.config.slow_ma_period).mean()
        df['MA_Slope'] = df['Functions_Slope'] = df['Fast_MA'].diff(5) # Simple slope proxy
        
        # --- RESTORED CLASSIC STATIONARY FEATURES (ALPHA HUNT V1) ---
        
        # 1. RSI (Relative Strength Index) - Mean Reversion
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['RSI'] = 100 - (100 / (1 + rs))
        # Normalize RSI to 0-1 or centered around 0? Let's use Z-score of RSI for ML
        df['RSI_Z'] = (df['RSI'] - 50) / 25.0
        
        # 2. TEMA Slope (Triple Exponential Moving Average) - Trend
        # TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
        ema1 = df['Close'].ewm(span=20, adjust=False).mean()
        ema2 = ema1.ewm(span=20, adjust=False).mean()
        ema3 = ema2.ewm(span=20, adjust=False).mean()
        df['TEMA'] = 3*ema1 - 3*ema2 + ema3
        df['TEMA_Slope'] = df['TEMA'].diff(3) / df['TEMA'] # Normalized slope

        # --- ADVANCED FEATURES (PHASE 1) ---
        # 1. Log Returns (Stationary)
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatility Ratio (Compression/Expansion)
        # Ratio of short-term ATR to long-term ATR
        atr_fast = tr.rolling(14).mean()
        atr_slow = tr.rolling(50).mean()
        df['Vol_Ratio'] = atr_fast / atr_slow.replace(0, np.nan)
        
        # 3. Trend Distance (Mean Reversion Pressure)
        # Normalized distance from Long Term MA
        ma_200 = df['Close'].rolling(200).mean()
        df['Trend_Dist'] = (df['Close'] - ma_200) / ma_200.replace(0, np.nan)
        
        # 4. Lagged Returns (Path Dependency)
        df['Ret_Lag1'] = df['Log_Returns'].shift(1)
        df['Ret_Lag2'] = df['Log_Returns'].shift(2)
        df['Ret_Lag3'] = df['Log_Returns'].shift(3)
        
        # 5. ADX (Average Directional Index) - Trend Strength
        # +DI = 100 * EMA(+DM) / ATR
        # -DI = 100 * EMA(-DM) / ATR
        # ADX = EMA(|+DI - -DI| / (+DI + -DI))
        adx_period = 14
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr_for_adx = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_adx = tr_for_adx.rolling(adx_period).mean()
        
        plus_di = 100 * plus_dm.rolling(adx_period).mean() / atr_adx.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(adx_period).mean() / atr_adx.replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['ADX'] = dx.rolling(adx_period).mean()
        
        # ADX Filter: Only trade when ADX > 25 (strong trend)
        df['ADX_Filter'] = (df['ADX'] > 25).astype(int)
        
        # 6. Volatility Regime (Numeric Encoding for ML)
        # Compute inline (same logic as RegimeEngine)
        vol = df['Volatility']
        vol_low = vol.rolling(100).quantile(self.config.vol_low_pct / 100)
        vol_high = vol.rolling(100).quantile(self.config.vol_high_pct / 100)
        df['Vol_Regime_Num'] = 1  # Default NORMAL
        df.loc[vol < vol_low, 'Vol_Regime_Num'] = 0  # LOW
        df.loc[vol > vol_high, 'Vol_Regime_Num'] = 2  # HIGH
        
        # --- ROBUSTNESS FEATURES (PHASE 2 EXPANSION) ---
        # A.1 Time & Session Structure
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek # 0=Mon, 6=Sun
        
        # Session Flags (UTC)
        # Asia: 00-08, London: 07-16, NY: 13-22
        df['Is_Asia'] = ((df['Hour'] >= 0) & (df['Hour'] < 8)).astype(int)
        df['Is_London'] = ((df['Hour'] >= 7) & (df['Hour'] < 16)).astype(int)
        df['Is_NY'] = ((df['Hour'] >= 13) & (df['Hour'] < 22)).astype(int)
        
        # Gap Flag
        # Calculate time delta in minutes
        delta_minutes = df.index.to_series().diff().dt.total_seconds() / 60.0
        df['DeltaMinutes'] = delta_minutes
        df['Gap_Flag'] = (delta_minutes > 70).astype(int) # Standard gap > 1h10m
        
        # A.2 Volatility / Tail-Shape Refinement
        # Downside vs Upside Volatility (50 period)
        r = df['Log_Returns']
        window_shape = 50
        downside = r.where(r < 0, 0.0)
        upside = r.where(r > 0, 0.0)
        df['Downside_Vol_50'] = downside.rolling(window_shape).std()
        df['Upside_Vol_50'] = upside.rolling(window_shape).std()
        
        # Rolling Skew / Kurtosis
        # Note: rolling().skew() available in newer pandas
        df['Skew_50'] = r.rolling(window_shape).skew()
        df['Kurt_50'] = r.rolling(window_shape).kurt()
        
        # Normalized Range
        df['RangeNorm'] = (df['High'] - df['Low']) / df['Close']
        
        # Asset Drawdown State (Distance from 200-period High)
        rolling_high = df['Close'].rolling(200).max()
        df['Asset_DD_200'] = (df['Close'] / rolling_high) - 1.0
        
        df = df.dropna()
        return df

    def add_features_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        print(f"\n[CALCULATING FEATURES]")
        print("-" * 40)
        processed = {}
        for sym, df in data.items():
            df_feat = self.add_features(df)
            processed[sym] = df_feat
            print(f"  {sym}: {len(df)} bars, {len(df_feat)} valid after indicators")
            
        print("[CALCULATING CROSS-SECTIONAL FEATURES]")
        processed = self.add_cross_sectional_features_all(processed)
        return processed

    def add_cross_sectional_features_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        # B. Cross-Sectional Features (Across Symbols)
        # Momentum Ranks (24h, 5d) and Vol Rank
        
        # 1. Create Pivot Tables
        # Need aligned index. Data is already aligned by DataLoader.
        
        # Extract Series
        mom_24h_dict = {}
        mom_5d_dict = {}
        vol_dict = {}
        
        # Define lookbacks
        idx_24h = 24
        idx_5d = 24 * 5
        
        for sym, df in data.items():
            # Recalculate Mom here to ensure we have it aligned? 
            # Or use existing? Existing 'Momentum' is lookback 10 (config).
            # We want specific 24h/5d here.
            mom_24h_dict[sym] = df['Close'] / df['Close'].shift(idx_24h) - 1
            mom_5d_dict[sym] = df['Close'] / df['Close'].shift(idx_5d) - 1
            vol_dict[sym] = df['Volatility']
            
        # Create DataFrames (Aligned)
        mom_24h_df = pd.DataFrame(mom_24h_dict)
        mom_5d_df = pd.DataFrame(mom_5d_dict)
        vol_df = pd.DataFrame(vol_dict)
        
        # Rank (pct=True)
        mom_24h_rank = mom_24h_df.rank(axis=1, pct=True)
        mom_5d_rank = mom_5d_df.rank(axis=1, pct=True)
        vol_rank = vol_df.rank(axis=1, pct=True)
        
        # Assign back
        processed = {}
        for sym, df in data.items():
            df = df.copy()
            # Map ranks back. Use index to align.
            # Using update or direct assignment
            if sym in mom_24h_rank.columns:
                 df['Mom_24h_rank'] = mom_24h_rank[sym]
                 df['Mom_5d_rank'] = mom_5d_rank[sym]
                 df['Vol_rank'] = vol_rank[sym]
            processed[sym] = df
            
        return processed

# ============================================================================
# REGIME DETECTION
# ============================================================================
class RegimeEngine:
    def __init__(self, config: Config):
        self.config = config
    
    def compute_vol_regime(self, df: pd.DataFrame) -> pd.Series:
        vol = df['Volatility']
        vol_low_threshold = vol.expanding().quantile(self.config.vol_low_pct / 100.0)
        vol_high_threshold = vol.expanding().quantile(self.config.vol_high_pct / 100.0)
        
        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = VolRegime.NORMAL.value
        regime[vol < vol_low_threshold] = VolRegime.LOW.value
        regime[vol > vol_high_threshold] = VolRegime.HIGH.value
        return regime

    def compute_trend_regime(self, df: pd.DataFrame) -> pd.Series:
        fast_ma = df['Fast_MA']
        slow_ma = df['Slow_MA']
        ma_slope = df['MA_Slope']
        
        ma_diff_pct = (fast_ma - slow_ma) / slow_ma
        
        regime = pd.Series(index=df.index, dtype=object)
        regime[:] = TrendRegime.RANGE.value
        
        bull_cond = (ma_diff_pct > self.config.trend_slope_threshold) & (ma_slope > 0)
        regime[bull_cond] = TrendRegime.BULL.value
        
        bear_cond = (ma_diff_pct < -self.config.trend_slope_threshold) & (ma_slope < 0)
        regime[bear_cond] = TrendRegime.BEAR.value
        return regime
        
    def add_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Vol_Regime'] = self.compute_vol_regime(df)
        df['Trend_Regime'] = self.compute_trend_regime(df)
        
        # A.3 Regime Duration Counters
        # "Time in Regime"
        # Reset counter when regime changes
        # Trend
        change_trend = df['Trend_Regime'] != df['Trend_Regime'].shift(1)
        # Cumsum identifies groups. Groupby count gets duration.
        # Efficient pandas way:
        df['Trend_Regime_Duration'] = df.groupby(change_trend.cumsum()).cumcount() + 1
        
        # Vol
        change_vol = df['Vol_Regime'] != df['Vol_Regime'].shift(1)
        df['Vol_Regime_Duration'] = df.groupby(change_vol.cumsum()).cumcount() + 1
        
        # Stub for HMM
        df['HMM_Regime'] = HMMRegime.UNKNOWN.value 
        return df

    def add_regimes_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed = {}
        for sym, df in data.items():
            processed[sym] = self.add_regimes(df)
        return processed

# ============================================================================
# ALPHA ENGINE (XGBOOST)
# ============================================================================
class AlphaEngine:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = None
        if self.config.use_alpha_engine:
            # v2.1 LEAN FEATURE SET (Advanced + Time + Lags ONLY)
            # RESTORED: Basic Stationary Features for Alpha Hunt v1
            self.feature_cols = [
                # Classic / Stationary
                'Z_Score', 'RSI_Z', 'TEMA_Slope', 'Vol_Intensity',
                # Advanced / Structural
                'Vol_Ratio', 'Trend_Dist', 'RangeNorm', 'Asset_DD_200', 
                'Trend_Regime_Duration', 'Vol_Regime_Duration', 'Efficiency_Ratio',
                'Downside_Vol_50', 'Upside_Vol_50', 'Skew_50', 'Kurt_50',
                # Lags
                'Ret_Lag1', 'Ret_Lag2', 'Ret_Lag3',
                # TimeMeta
                'Hour', 'DayOfWeek', 'Is_Asia', 'Is_London', 'Is_NY'
            ]
        else:
            self.feature_cols = []
        
        self.trained_symbols = []

    def _create_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df_copy = df.copy()
        
        df_copy['Future_Return'] = df_copy['Close'].pct_change(self.config.alpha_target_lookahead).shift(-self.config.alpha_target_lookahead)
        
        df_copy['Target'] = 0
        df_copy.loc[df_copy['Future_Return'] > self.config.alpha_return_threshold, 'Target'] = 1
        df_copy.loc[df_copy['Future_Return'] < -self.config.alpha_return_threshold, 'Target'] = -1
        
        df_copy = df_copy.dropna(subset=self.feature_cols + ['Target'])
        
        X = df_copy[self.feature_cols]
        y = df_copy['Target'] + 1
        return X, y

    def train_model(self, data: Dict[str, pd.DataFrame]):
        print(f"\n[TRAINING ALPHA ENGINE (XGBOOST)]")
        print("-" * 40)
        all_X = []
        all_y = []
        
        for sym, df in data.items():
            try:
                # Rolling Split Logic (80/20)
                split_idx = int(len(df) * self.config.ml_train_split_pct)
                if split_idx < 100: # Insufficient data
                    continue
                    
                df_train = df.iloc[:split_idx]
                
                X, y = self._create_features_and_target(df_train)
                all_X.append(X)
                all_y.append(y)
                self.trained_symbols.append(sym)
            except Exception as e:
                print(f"Skipping {sym}: {e}")
        
        if not all_X:
            print("No training data available.")
            return
            
        X_train = pd.concat(all_X)
        y_train = pd.concat(all_y)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.config.alpha_model_params, dtrain, num_boost_round=100)
        print(f"  Alpha Engine trained on {len(self.trained_symbols)} symbols.")

    def generate_signals_with_probs(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        if self.model is None:
            # Return empty structure with correct columns to prevent KeyError
            dummy_probs = pd.DataFrame(0.0, columns=['prob_down', 'prob_neural', 'prob_up', 'prob_margin', 'prob_entropy'], index=df.index)
            return pd.Series(0, index=df.index), dummy_probs
            
        X = df[self.feature_cols]
        dtest = xgb.DMatrix(X)
        
        probs = self.model.predict(dtest)
        
        # probs is [n_samples, 3] (Down, Neutral, Up)
        prob_df = pd.DataFrame(probs, columns=['prob_down', 'prob_neural', 'prob_up'], index=df.index)
        
        # C. ML Signal-Quality Meta-Features
        # 11. Probability Margin (Max - Neutral) (Confidence in Direction)
        prob_df['prob_max'] = prob_df[['prob_up', 'prob_down']].max(axis=1)
        prob_df['prob_margin'] = prob_df['prob_max'] - prob_df['prob_neural']
        
        # 12. Prediction Entropy (Uncertainty)
        # H = -sum(p * log(p))
        eps = 1e-12
        p = prob_df[['prob_down', 'prob_neural', 'prob_up']].clip(eps, 1.0)
        prob_df['prob_entropy'] = - (p * np.log(p)).sum(axis=1)
        
        signal = pd.Series(0, index=df.index)
        signal[prob_df['prob_up'] > self.config.ml_prob_threshold_long] = 1
        signal[prob_df['prob_down'] > self.config.ml_prob_threshold_short] = -1
        
        return signal, prob_df

    def train_predict_walk_forward(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Strict Walk-Forward Optimization to prevent data leakage.
        1. Aligns data.
        2. Loops through time with Rolling Window.
        3. Trains on [t-TrainSz : t], Predicts on [t : t+TestSz].
        4. Stitches OOS predictions.
        """
        print(f"\n[ALPHA ENGINE] Starting Walk-Forward Optimization (Train={self.config.wfo_train_bars}, Test={self.config.wfo_test_bars})...")
        
        # 1. Get Global Time Index
        all_indices = sorted(list(set().union(*[df.index for df in data.values()])))
        full_time_index = pd.to_datetime(all_indices).sort_values()
        
        if len(full_time_index) < self.config.wfo_train_bars + self.config.wfo_test_bars:
            print("[ERROR] Insufficient data for even one WFO fold.")
            return data

        # 2. Initialize Result Containers
        oos_predictions = {sym: [] for sym in data.keys()}
        
        # 3. Walk-Forward Loop
        start_idx = 0
        current_idx = self.config.wfo_train_bars
        
        total_bars = len(full_time_index)
        
        while current_idx < total_bars:
            train_end_idx = current_idx
            test_end_idx = min(current_idx + self.config.wfo_test_bars, total_bars)
            
            train_start_time = full_time_index[max(0, train_end_idx - self.config.wfo_train_bars)]
            train_end_time = full_time_index[train_end_idx-1]
            test_start_time = full_time_index[train_end_idx]
            test_end_time = full_time_index[test_end_idx-1]
            
            print(f"  > Fold: Train[{train_start_time.date()} : {train_end_time.date()}] -> Test[{test_start_time.date()} : {test_end_time.date()}]")
            
            # A. Prepare Training Data (Global)
            X_train_list = []
            y_train_list = []
            
            for sym, df in data.items():
                # Slice training window
                mask_train = (df.index >= train_start_time) & (df.index <= train_end_time)
                df_train = df.loc[mask_train]
                
                if len(df_train) < 50: continue
                
                X, y = self._create_features_and_target(df_train)
                X_train_list.append(X)
                y_train_list.append(y)
                
            if not X_train_list:
                print("    [WARN] No training data for this fold.")
                current_idx += self.config.wfo_test_bars
                continue
                
            X_train = pd.concat(X_train_list)
            y_train = pd.concat(y_train_list)
            
            # B. Train Model
            dtrain = xgb.DMatrix(X_train, label=y_train)
            model = xgb.train(self.config.alpha_model_params, dtrain, num_boost_round=100)
            
            # C. Predict (Test Window)
            for sym, df in data.items():
                mask_test = (df.index >= test_start_time) & (df.index <= test_end_time)
                df_test = df.loc[mask_test]
                
                if df_test.empty:
                    continue
                    
                X_test = df_test[self.feature_cols]
                dtest_xgb = xgb.DMatrix(X_test)
                probs = model.predict(dtest_xgb)
                
                # Create mini-dataframe for this OOS chunk
                # probs is [n_samples, 3]
                cols = ['prob_down', 'prob_neural', 'prob_up']
                chunk_df = pd.DataFrame(probs, columns=cols, index=df_test.index)
                
                oos_predictions[sym].append(chunk_df)

            # Slide Window
            current_idx += self.config.wfo_test_bars
            
        # 4. Stitch and Merge into Data
        processed = {}
        for sym, df in data.items():
            df_out = df.copy()
            
            # Default to 0/Neutral for periods before first prediction
            df_out['prob_up'] = 0.0
            df_out['prob_down'] = 0.0
            df_out['prob_neural'] = 1.0 # Default neutral
            df_out['S_Alpha'] = 0
            
            if oos_predictions[sym]:
                full_oos = pd.concat(oos_predictions[sym])
                # Join indices
                common_idx = df_out.index.intersection(full_oos.index)
                
                df_out.loc[common_idx, 'prob_up'] = full_oos.loc[common_idx, 'prob_up']
                df_out.loc[common_idx, 'prob_down'] = full_oos.loc[common_idx, 'prob_down']
                df_out.loc[common_idx, 'prob_neural'] = full_oos.loc[common_idx, 'prob_neural']
                
                # Compute Derived Metrics for Entry Logic
                p_up = df_out['prob_up']
                p_down = df_out['prob_down']
                p_neu = df_out['prob_neural']
                
                # 1. Probability Margin (Confidence difference)
                # How much more likely is the dominant direction than the opposite?
                df_out['prob_margin'] = abs(p_up - p_down)
                
                # 2. Probability Max (For sizing)
                df_out['prob_max'] = df_out[['prob_up', 'prob_down']].max(axis=1)
                
                # 3. Entropy (Uncertainty) - Optional context
                # H = -sum(p * log2(p))
                # Clip to avoid log(0)
                epsilon = 1e-9
                p_up_c = p_up.clip(epsilon, 1.0)
                p_down_c = p_down.clip(epsilon, 1.0)
                p_neu_c = p_neu.clip(epsilon, 1.0)
                
                entropy = -(p_up_c * np.log2(p_up_c) + 
                           p_down_c * np.log2(p_down_c) + 
                           p_neu_c * np.log2(p_neu_c))
                df_out['prob_entropy'] = entropy
                
            processed[sym] = df_out
            
        # 5. Apply Rank Logic (Cross-Sectional)
        if hasattr(self.config, 'use_rank_logic') and self.config.use_rank_logic:
            processed = self.add_rank_signals(processed)
        else:
             # Standard Threshold Logic
             for sym, df in processed.items():
                 signal = pd.Series(0, index=df.index)
                 signal[df['prob_up'] > self.config.ml_prob_threshold_long] = 1
                 signal[df['prob_down'] > self.config.ml_prob_threshold_short] = -1
                 df['S_Alpha'] = signal
                 processed[sym] = df
                 
        return processed

    def add_rank_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Cross-Sectional Ranking Strategy (Alpha V3).
        Forces trades by picking Top N and Bottom N assets per timestamp.
        """
        print(f"\n[RANK SIGNAL] Generating Top {self.config.rank_top_n} / Bottom {self.config.rank_top_n} Signals...")
        
        # 1. Pivot probs to [Timestamp x Symbol] matrix
        # Be careful with mismatched indices. We rely on time alignment.
        all_probs_up = {}
        all_probs_down = {}
        
        for sym, df in data.items():
            if 'prob_up' in df.columns:
                all_probs_up[sym] = df['prob_up']
                all_probs_down[sym] = df['prob_down']
                
        df_up = pd.DataFrame(all_probs_up)
        # df_down = pd.DataFrame(all_probs_down) # Not strictly needed if we rank by Prob_Up
        
        # 2. Compute Ranks (Descending: High Prob Up = Rank 1)
        # method='min': ties get same rank
        ranks = df_up.rank(axis=1, ascending=False, method='min')
        
        # 3. Compute Reverse Ranks (Ascending: Low Prob Up = Rank 1 = High Prob Down?)
        # Actually, let's use Prob_Down for shorts to be precise, or just flip Prob_Up?
        # Let's use Prob_Down for Short Ranking to be precise.
        df_down = pd.DataFrame(all_probs_down)
        ranks_down = df_down.rank(axis=1, ascending=False, method='min') # Rank 1 = Highest Prob Down
        
        # 4. Generate Signals
        for sym, df in data.items():
            if sym not in ranks.columns:
                continue
                
            signal = pd.Series(0, index=df.index)
            
            # Long Top N (Highest Prob Up)
            signal[ranks[sym] <= self.config.rank_top_n] = 1
            
            # Short Top N (Highest Prob Down)
            # Ensure no conflict (rare)
            signal[ranks_down[sym] <= self.config.rank_top_n] = -1
            
            df['S_Alpha'] = signal
            data[sym] = df
            
        return data

    def add_signals_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if not self.config.use_alpha_engine:
            return data
            
        # LIVE MODE: Train on everything, predict on everything (Inference)
        # OR if Model is already trained (from generic Backtest loop), just predict (Inference)
        if (hasattr(self.config, 'mode') and self.config.mode in ["LIVE", "PAPER", "SHADOW"]) or (self.model is not None):
             print(f"\n[GENERATING ALPHA SIGNALS] (Mode: {self.config.mode} - Inference)")
             if self.model is None: # Only train if not trained
                 self.train_model(data) 
             
             processed = {}
             for sym, df in data.items():
                 df_sig = df.copy()
                 signal, probs = self.generate_signals_with_probs(df)
                 
                 # Placeholder signal (will be overwritten if Rank Logic is on)
                 df_sig['S_Alpha'] = signal
                 df_sig['prob_up'] = probs['prob_up']
                 df_sig['prob_down'] = probs['prob_down']
                 df_sig['prob_neural'] = probs['prob_neural']
                 
                 # Compute Derived Metrics (Consistency with WFO)
                 p_up = df_sig['prob_up']
                 p_down = df_sig['prob_down']
                 p_neu = df_sig['prob_neural']
                 
                 df_sig['prob_margin'] = abs(p_up - p_down)
                 df_sig['prob_max'] = df_sig[['prob_up', 'prob_down']].max(axis=1)
                 
                 epsilon = 1e-9
                 p_up_c = p_up.clip(epsilon, 1.0)
                 p_down_c = p_down.clip(epsilon, 1.0)
                 p_neu_c = p_neu.clip(epsilon, 1.0)
                 
                 entropy = -(p_up_c * np.log2(p_up_c) + 
                            p_down_c * np.log2(p_down_c) + 
                            p_neu_c * np.log2(p_neu_c))
                 df_sig['prob_entropy'] = entropy
                 
                 processed[sym] = df_sig
             
             # Apply Rank Logic for Live Mode too
             if hasattr(self.config, 'use_rank_logic') and self.config.use_rank_logic:
                 processed = self.add_rank_signals(processed)
                 
             return processed
        
        # BACKTEST MODE: Walk-Forward Optimization
        else:
             return self.train_predict_walk_forward(data)

# ============================================================================
# SIGNAL ENSEMBLE
# ============================================================================
class EnsembleSignal:
    def __init__(self, config: Config):
        self.config = config

    def compute_ensemble(self, df: pd.DataFrame) -> pd.Series:
        # Static Weights (Phase 1)
        # We assume ML (S_Alpha) is present
        
        ensemble_score = pd.Series(0.0, index=df.index)
        
        if 'S_Alpha' in df.columns:
            ensemble_score += self.config.w_alpha * df['S_Alpha']
            
        # Ignoring other signals for Phase 1 as weights are 0.0 in config
        # But for completeness:
        # ensemble_score += self.config.w_mr * df.get('S_MR', 0)
        
        # StatArb (if present)
        if 'StatArb_Signal' in df.columns:
            ensemble_score += self.config.w_stat_arb * df['StatArb_Signal']
        
        return ensemble_score

    def add_ensemble_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        print(f"\n[COMPUTING ENSEMBLE SIGNALS]")
        processed = {}
        for sym, df in data.items():
            df['Ensemble_Score'] = self.compute_ensemble(df)
            processed[sym] = df
        return processed

# ============================================================================
# CRISIS ALPHA ENGINE (MEAN-REVERSION DURING HIGH VOLATILITY)
# ============================================================================
class CrisisAlphaEngine:
    """
    Regime-switching crisis module.
    - Normal regime: Uses standard momentum/ML signals
    - Crisis regime: Switches to mean-reversion (counter-trend)
    
    Theory: During crises, momentum fails (stop-outs cascade, reversals occur).
    Mean-reversion capitalizes on overshoots and snap-backs.
    """
    
    def __init__(self, config: Config):
        self.config = config
        # Crisis detection thresholds
        self.vol_expansion_threshold = 2.0  # ATR > 2x historical = crisis
        self.z_score_entry = 2.0  # Enter counter-trend at 2σ deviation
        self.z_score_exit = 0.5   # Exit when price returns to mean
        
    def detect_crisis(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect crisis regime using volatility expansion.
        Crisis = current ATR > vol_expansion_threshold × rolling ATR mean
        """
        if 'ATR' not in df.columns:
            return pd.Series(False, index=df.index)
        
        atr = df['ATR']
        atr_ma = atr.rolling(100).mean()  # Long-term ATR average
        vol_ratio = atr / atr_ma.replace(0, np.nan)
        
        is_crisis = vol_ratio > self.vol_expansion_threshold
        return is_crisis
    
    def generate_crisis_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Mean-reversion signals for crisis regime.
        - Long when Z-score < -2 (oversold)
        - Short when Z-score > +2 (overbought)
        - Exit at Z-score ~ 0
        """
        if 'Z_Score' not in df.columns:
            return pd.Series(0, index=df.index)
        
        z = df['Z_Score']
        
        signal = pd.Series(0, index=df.index)
        signal[z < -self.z_score_entry] = 1   # Oversold → Long
        signal[z > self.z_score_entry] = -1   # Overbought → Short
        
        return signal
    
    def add_crisis_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Iteration 4: Kelly-Optimal Continuous Sizing
        
        Position size = 1 / (vol_ratio)^2
        
        This is mathematically optimal because:
        - Kelly criterion suggests f* ∝ edge / variance
        - When volatility doubles, optimal size should quarter (inverse square)
        - Provides smooth transition, not binary jumps
        """
        print(f"\n[CRISIS ALPHA ENGINE - KELLY SIZING]")
        
        processed = {}
        
        for sym, df in data.items():
            df = df.copy()
            
            # Calculate volatility ratio
            atr = df.get('ATR', pd.Series(1, index=df.index))
            atr_ma = atr.rolling(100).mean()
            vol_ratio = (atr / atr_ma.replace(0, np.nan)).fillna(1.0)
            
            # Continuous Kelly-optimal sizing: inverse square of volatility ratio
            # f* = 1 / vol_ratio^2, capped at [0.1, 1.0]
            kelly_size = 1 / (vol_ratio ** 2)
            kelly_size = kelly_size.clip(0.1, 1.0)  # Min 10%, Max 100% of normal size
            
            df['Crisis_Size_Mult'] = kelly_size
            
            # Final signal = Ensemble × Kelly Size
            df['Final_Signal'] = df['Ensemble_Score'] * df['Crisis_Size_Mult']
            
            processed[sym] = df
        
        avg_kelly = np.mean([df['Crisis_Size_Mult'].mean() for df in processed.values()])
        print(f"  Average Kelly Size Multiplier: {avg_kelly:.2f}x")
        print(f"  Formula: size = 1 / (vol_ratio)^2, capped [0.1, 1.0]")
        print(f"  Safe-Haven: Gold boosted +0.5 signal when vol > 1.5x")
        return processed

# ============================================================================
# STAT ARB ENGINE (PAIRS TRADING)
# ============================================================================
class StatArbEngine:
    def __init__(self, config: Config):
        self.config = config
        self.pairs = [] # List of tuples (sym1, sym2, hedge_ratio, spread_mean, spread_std)
        self.trained = False

    def fit_ou_process(self, spread: pd.Series):
        """
        Fits an Ornstein-Uhlenbeck process to the spread:
        dX_t = theta * (mu - X_t) * dt + sigma * dW_t
        Returns: theta (mean rev speed), mu (long term mean), sigma (vol)
        """
        dt = 1.0 / 252.0 # Annualized
        
        X_t = spread.values[:-1]
        X_tp1 = spread.values[1:]
        
        # Regress X_{t+1} on X_t
        # X_{t+1} = a * X_t + b + epsilon
        # theta = -ln(a) / dt
        # mu = b / (1-a)
        # sigma = std(eps) * sqrt( -2ln(a) / (dt*(1-a^2)) )
        
        X = X_t.reshape(-1, 1)
        y = X_tp1
        
        lr = LinearRegression()
        lr.fit(X, y)
        
        a = lr.coef_[0]
        b = lr.intercept_
        
        # Stability check
        if a >= 1.0: # Non-stationary or explosive
            return 0.0, 0.0, 0.0
            
        theta = -np.log(a) / dt
        mu = b / (1 - a)
        
        epsilon = y - lr.predict(X)
        sigma_eps = np.std(epsilon)
        
        # Exact sigma derivation from variance of AR(1) residual
        # Var(eps) = sigma^2 * (1 - exp(-2*theta*dt)) / (2*theta)
        # Approximate for small dt: sigma_eps = sigma * sqrt(dt)
        sigma = sigma_eps / np.sqrt(dt) 
        
        return theta, mu, sigma

    def find_pairs(self, data: Dict[str, pd.DataFrame]):
        """
        Identify cointegrated pairs using OU Process parameters.
        Selects pairs with high Mean Reversion Speed (theta).
        """
        print(f"\n[STAT ARB] Scanning for Pairs (OU Process)...")
        symbols = sorted(list(data.keys()))
        prices = pd.DataFrame({sym: data[sym]['Close'] for sym in symbols}).dropna()
        
        corr_matrix = prices.corr()
        candidates = []
        
        # 1. Broad Correlation Filter
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1 = symbols[i]
                s2 = symbols[j]
                if corr_matrix.loc[s1, s2] > 0.7:
                    candidates.append((s1, s2))
        
        print(f"  Candidates: {len(candidates)}")
        
        found_pairs = []
        for s1, s2 in candidates:
            # Regress S1 on S2 to get Hedge Ratio (Beta)
            y = prices[s1].values.reshape(-1, 1)
            x = prices[s2].values.reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x, y)
            beta = lr.coef_[0][0]
            
            # Form the Spread
            spread = prices[s1] - beta * prices[s2]
            
            # Fit OU Process
            theta, mu, sigma = self.fit_ou_process(spread)
            
            # Filter: Theta > 10 means 1/10th of a year (25 days) half-life? 
            # Half-life = ln(2)/theta. 
            # If theta=10, HL = 0.07 years = 17 days. 
            # If theta=252, HL = 1 day.
            # We want fast reversion.
            
            if theta > 5.0: # Only fast mean reverters
                # Avellaneda & Lee Optimal Entry Threshold
                # s_entry = sigma / sqrt(2*theta) assuming zero transaction costs, or similar
                # We store parameters to compute dynamic thresholds
                
                print(f"    Found Pair: {s1} vs {s2} (Beta={beta:.2f}, Theta={theta:.2f}, Mu={mu:.4f}, Sigma={sigma:.4f})")
                found_pairs.append({
                    's1': s1, 's2': s2, 'beta': beta,
                    'theta': theta, 'mu': mu, 'sigma': sigma
                })
        
        self.pairs = found_pairs
        self.trained = True

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        data_out = {sym: df.copy() for sym, df in data.items()}
        for sym in data_out:
            data_out[sym]['StatArb_Signal'] = 0.0
            
        if not self.trained or not self.pairs:
            return data_out
            
        print(f"\n[STAT ARB] Generating Signals (OU Logic)...")
        
        for pair in self.pairs:
            s1, s2 = pair['s1'], pair['s2']
            beta = pair['beta']
            theta = pair['theta']
            mu = pair['mu']
            sigma = pair['sigma']
            
            if s1 not in data_out or s2 not in data_out: continue
            
            combined_idx = data_out[s1].index.intersection(data_out[s2].index)
            p1 = data_out[s1].loc[combined_idx, 'Close']
            p2 = data_out[s2].loc[combined_idx, 'Close']
            
            # Calculate Spread
            spread = p1 - beta * p2
            
            # Optimal Entry Threshold (Avellaneda & Lee)
            # Entry when spread deviates by sigma_eq
            # sigma_eq = sigma / sqrt(2*theta)
            # Let's target 1.5 * Sigma_Eq for safety
            
            sigma_eq = sigma / np.sqrt(2 * theta)
            entry_threshold = 2.0 * sigma_eq # Strict entry
            exit_threshold = 0.0 # Revert to mean
            
            # Recenter spread around mu
            spread_centered = spread - mu
            
            sig_s1 = pd.Series(0.0, index=combined_idx)
            sig_s2 = pd.Series(0.0, index=combined_idx)
            
            # Long Spread (Buy S1, Sell S2) when spread < -threshold
            long_cond = spread_centered < -entry_threshold
            sig_s1[long_cond] = 1.0
            sig_s2[long_cond] = -1.0
            
            # Short Spread (Sell S1, Buy S2) when spread > entry_threshold
            short_cond = spread_centered > entry_threshold
            sig_s1[short_cond] = -1.0
            sig_s2[short_cond] = 1.0
            
            # Add to dataframe
            data_out[s1].loc[combined_idx, 'StatArb_Signal'] += sig_s1
            data_out[s2].loc[combined_idx, 'StatArb_Signal'] += sig_s2
            
        return data_out



# ============================================================================
# ACCOUNT & POSITION TRACKING
# ============================================================================
@dataclass
class Position:
    symbol: str
    direction: int # 1 or -1
    entry_price: float
    entry_time: pd.Timestamp
    size: float
    sl: float
    tp: float
    entry_signal_score: float
    entry_context: dict # To store Regime, Signal Score, etc.

    # Validation Fields
    mae: float = 0.0 # Max Adverse Excursion (Price Dist)
    mfe: float = 0.0 # Max Favorable Excursion (Price Dist)
    highest_price: float = 0.0
    lowest_price: float = float('inf')
    
    def __post_init__(self):
        if self.highest_price == 0.0: self.highest_price = self.entry_price
        if self.lowest_price == float('inf'): self.lowest_price = self.entry_price

    def update_extremes(self, high, low):
        # Update Price Extremes
        if high > self.highest_price: self.highest_price = high
        if low < self.lowest_price: self.lowest_price = low
        
        # Calculate MAE/MFE based on direction
        if self.direction == 1: # LONG
            # MAE = Entry - Lowest Low (Positive value of loss distance)
            self.mae = max(self.mae, self.entry_price - self.lowest_price)
            # MFE = Highest High - Entry
            self.mfe = max(self.mfe, self.highest_price - self.entry_price)
        else: # SHORT
            # MAE = Highest High - Entry
            self.mae = max(self.mae, self.highest_price - self.entry_price)
            # MFE = Entry - Lowest Low
            self.mfe = max(self.mfe, self.entry_price - self.lowest_price)

class Account:
    def __init__(self, config: Config):
        self.config = config
        self.balance = config.initial_balance
        self.equity = config.initial_balance
        self.peak_equity = config.initial_balance # Track High Water Mark
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        
        # Streak Tracking
        self.current_streak_len = 0
        self.current_streak_sign = 0 # 1=Win, -1=Loss
        
    def calculate_position_size(self, price: float, stop_loss_dist: float, prob_win: float = 0.5) -> float: # Added prob_win
        # Kelly Criterion
        # f = p - (1-p)/b
        # We assume b (R-Multiple) ~ 1.2 (Conservative estimate from backtest)
        # To be safe, use b=1.0 for calculation (Under-betting is safer).
        
        if stop_loss_dist <= 0: return 0.0
        
        b = 1.5 # Adjusted R-Multiple (Strategy Target is >1.5)
        kelly_fraction = prob_win - (1 - prob_win) / b
        
        # Debug small sample
        # if np.random.rand() < 0.001:
        #    print(f"[KELLY DEBUG] P={prob_win:.2f}, b={b}, K={kelly_fraction:.2f}")

        # Half-Kelly for safety
        safe_fraction = kelly_fraction * 0.5
        
        if safe_fraction < 0: safe_fraction = 0.0
        
        # Dynamic Risk Limit
        max_risk_pct = self.config.risk_per_trade # 0.8%
        
        # Final Risk % to use
        # If High Confidence (P=0.6) -> Kelly ~ 0.2 -> Half ~ 0.1 -> Cap at 0.008
        # Wait, Kelly often suggests HUGE size (10-20%).
        # So essentially this becomes "Trade Full Size unless P is very low".
        # Or should we scale relative to the Limit?
        # Standard interpretation: "Risk Limit is the MAX".
        # We use Min(Limit, Kelly).
        
        used_risk_pct = min(max_risk_pct, safe_fraction)
        
        # Also, scale down for low probability?
        # If P is just above 0.5 (e.g. 0.51), Kelly is tiny.
        # This gives us the "Conviction Sizing" automatically.
        
        risk_amount = self.equity * used_risk_pct
        
        if stop_loss_dist <= 0: return 0.0
        size = risk_amount / stop_loss_dist
        max_size = (self.equity * self.config.account_leverage) / price
        return min(size, max_size)

    def close_position(self, pos: Position, exit_price: float, exit_time: pd.Timestamp, reason: str):
        gross_pnl = (exit_price - pos.entry_price) * pos.size * pos.direction
        
        # === REALISTIC TRANSACTION COST MODEL ===
        # Audited 2024-12-09 for accuracy
        
        if '=X' in pos.symbol:  # FX pairs
            # pos.size is in UNITS (e.g., 300,000 units)
            # pip_value is $10 per pip for 1 STANDARD LOT (100,000 units)
            lots = pos.size / 100_000
            pip_value = 10.0  # $10 per pip for standard lot
            spread_pips = 1.0  # Conservative 1 pip spread (FTMO typically 0.5-1.5)
            cost = lots * pip_value * spread_pips * 2  # Entry + Exit
            
        elif '-USD' in pos.symbol:  # Crypto
            # Crypto: ~0.1% spread for retail (FTMO typically 0.05-0.1%)
            notional = pos.size * exit_price
            spread_pct = 0.001  # 0.1% spread
            cost = notional * spread_pct * 2  # Round-trip
            
        elif pos.symbol in ['ES=F', 'NQ=F', 'YM=F', 'RTY=F']:  # Index CFDs
            # Modeled as CFDs for Prop Firms (Spread based, no commission)
            # Spreads: US500 ~0.4, US100 ~1.5, US30 ~2.5, US2000 ~0.6
            spread_map = {
                'ES=F': 0.50,  # S&P 500
                'NQ=F': 1.60,  # Nasdaq 100
                'YM=F': 2.80,  # Dow 30
                'RTY=F': 0.70  # Russell 2000
            }
            spread_pts = spread_map.get(pos.symbol, 1.0)
            
            # Point Value assumption: 
            # If pos.size is Contracts (Unit Size), we multiply by Point Value multiplier?
            # Or is pos.size Dollar Value? Or Shares?
            # In sim, pos.size = Risk amount / Stop distance.
            # Usually 'Contracts'.
            # Futures Multipliers: ES=50, NQ=20, YM=5, RTY=50.
            # CFD Multipliers often 1, 10, or 20. 
            # Let's assume Standard Futures Multipliers as worst case?
            # Actually, standard CFD is often 1:1 or 1:10.
            # But earlier code used tick_values implies Multipliers.
            # Let's stick to 'tick_values' for PnL calculation implication.
            # If PnL uses multipliers, Cost must too.
            # Assuming Backtester PnL calculation accounts for multipliers somewhere?
            # Backtester `close_position` line 1317: (exit - entry) * pos.size * direction.
            # Wait. If pos.size is "Contracts" AND "Multiplier" is missing here, then PnL is raw points * contracts.
            # For NQ to lose -16%, it must match price.
            # If Price 18000. 1% move = 180 pts.
            # If Size 1. PnL = 180.
            # Real NQ Future: 180 pts * $20 = $3600.
            # If we trade "1 Contract" in this code, we get $180.
            # So we are trading "Mini-CFDs" (1:1).
            # Cost should be: Contracts * Spread * 1.0.
            
            contracts = abs(pos.size)
            cost = contracts * spread_pts * 1.0 # 1:1 Multiplier Assumption
            
        elif pos.symbol in ['GC=F', 'CL=F', 'NG=F']:  # Commodity Futures
            # Gold: tick=$10 (0.10 pts), Oil: tick=$10 (0.01 pts), Gas: tick=$10
            # Commodities have wider spreads, especially NG
            contracts = max(1, abs(pos.size / 10))  # Rough contract estimate
            tick_values = {'GC=F': 10.0, 'CL=F': 10.0, 'NG=F': 10.0}
            tick_val = tick_values.get(pos.symbol, 10.0)
            spread_ticks = 2.0 if pos.symbol == 'NG=F' else 1.0  # NG has wider spread
            commission_per_side = 2.50
            cost = contracts * (commission_per_side * 2 + spread_ticks * tick_val * 2)
            
        else:
            # Default fallback: 0.1% of notional (conservative)
            notional = abs(pos.size * exit_price)
            cost = notional * 0.001 * 2
        
        net_pnl = gross_pnl - cost
        
        self.balance += net_pnl
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance
            
        self.positions.remove(pos)
        
        # Streak Logic
        if net_pnl > 0:
            if self.current_streak_sign >= 0:
                self.current_streak_len += 1
                self.current_streak_sign = 1
            else:
                self.current_streak_len = 1
                self.current_streak_sign = 1
        elif net_pnl < 0:
            if self.current_streak_sign <= 0:
                self.current_streak_len += 1
                self.current_streak_sign = -1
            else:
                self.current_streak_len = 1
                self.current_streak_sign = -1
        
        # D. Trade-Level Meta-Features (Context)
        # Calculate R-Multiple based on ACTUAL entry risk
        risk_per_price = abs(pos.entry_price - pos.sl)
        entry_risk_dollars = risk_per_price * pos.size
        
        r_multiple = 0.0
        if entry_risk_dollars > 0:
            r_multiple = net_pnl / entry_risk_dollars
            
        trade_record = {
            'Symbol': pos.symbol,
            'Direction': 'LONG' if pos.direction == 1 else 'SHORT',
            'Entry Time': pos.entry_time,
            'Exit Time': exit_time,
            'Entry Price': pos.entry_price,
            'Exit Price': exit_price,
            'Size': pos.size,
            'PnL': net_pnl,
            'Reason': reason,
            'Balance': self.balance,
            
            # --- Robustness Metadata ---
            'Risk_Dollars': entry_risk_dollars,
            'R_Multiple': r_multiple,
            'Streak_Len_Exit': self.current_streak_len * self.current_streak_sign, # +/- Length
            'MAE': pos.mae,
            'MFE': pos.mfe,
            
            # Unpack Context
            **pos.entry_context
        }
        self.trade_history.append(trade_record)

def analyze_trades(trades: List[Dict]):
    if not trades:
        print("No trades to analyze.")
        return
        
    df = pd.DataFrame(trades)
    print("\n[DIAGNOSTICS] R-Multiple Analysis")
    print("-" * 40)
    
    # 1. By Trend Regime
    if 'Entry_Trend_Regime' in df.columns:
        print("\nBy Trend Regime:")
        print(df.groupby('Entry_Trend_Regime')['R_Multiple'].agg(['count', 'mean', 'median', 'sum']))
        
    # 2. By Vol Regime
    if 'Entry_Vol_Regime' in df.columns:
        print("\nBy Vol Regime:")
        print(df.groupby('Entry_Vol_Regime')['R_Multiple'].agg(['count', 'mean', 'median', 'sum']))
        
    # 3. By Vol Rank (Quintiles)
    if 'Entry_Vol_Rank' in df.columns:
        df['Vol_Rank_Bucket'] = pd.qcut(df['Entry_Vol_Rank'], 5, labels=False, duplicates='drop')
        print("\nBy Vol Rank (Quintiles 0=Low, 4=High):")
        print(df.groupby('Vol_Rank_Bucket')['R_Multiple'].agg(['count', 'mean', 'median']))

    # 4. By Prob Margin (Confidence)
    if 'Entry_Prob_Margin' in df.columns:
        try:
            df['Prob_Bucket'] = pd.qcut(df['Entry_Prob_Margin'], 5, labels=False, duplicates='drop')
            print("\nBy Confidence (Prob Margin Quintiles):")
            print(df.groupby('Prob_Bucket')['R_Multiple'].agg(['count', 'mean', 'median']))
        except ValueError:
            pass # Not enough unique values

class Backtester:
    def __init__(self, config: Config):
        self.config = config
        self.account = Account(config)
        self.pending_orders = {} # Sym -> Order Dict
    
    def run_backtest(self, data: Dict[str, pd.DataFrame]):
        # ... (setup code unchanged until loop) ...
        print(f"\n[RUNNING EVENT-BASED BACKTEST]")
        print("-" * 40)
        
        # 1. Align all data to a single timeline
        combined_index = pd.Index([])
        for df in data.values():
            combined_index = combined_index.union(df.index)
        combined_index = combined_index.sort_values()
        
        print(f"  Simulation Range: {combined_index[0]} -> {combined_index[-1]}")
        print(f"  Total Bars: {len(combined_index)}")
        
        equity_curve = []
        daily_start_equity = self.config.initial_balance
        current_day = None
        trading_blocked_today = False
        last_limit_hit_date = None
        
        for current_time in combined_index:
            # 0. EXECUTE PENDING ORDERS (NEXT OPEN)
            # This eliminates Lookahead Bias by trading at the Open of the bar,
            # using signals generated at the Close of the previous bar.
            executed_syms = []
            for sym, order in self.pending_orders.items():
                if sym not in data: continue
                if current_time not in data[sym].index: continue
                
                # Execute at Open
                row = data[sym].loc[current_time]
                open_price = row['Open']
                
                # Recalculate size (optional, but safer to use t-1 ATR for consistency)
                # We use the size calculated at t-1 logic or recalc?
                # Let's use logic: intended risk / (sl_dist at t-1).
                # But price changed from Close(t-1) to Open(t).
                # Re-eval size to ensure risk is constant $ (Vol-Adjusted)
                
                sl_dist = order['sl_dist'] # From t-1
                prob_win = order['prob_win']
                
                # New Size based on Open Price
                size = self.account.calculate_position_size(open_price, sl_dist, prob_win=prob_win)
                
                # Apply Penalty Factor if passed
                if order.get('penalty_applied', False):
                     size *= self.config.risk_penalty_factor
                
                if size <= 0: 
                    executed_syms.append(sym)
                    continue
                
                # Create Position
                new_pos = Position(
                    symbol=sym,
                    direction=order['direction'],
                    entry_price=open_price,
                    entry_time=current_time,
                    size=size,
                    sl=open_price - sl_dist if order['direction']==1 else open_price + sl_dist,
                    tp=0.0,
                    entry_signal_score=order['score'],
                    entry_context=order['context']
                )
                self.account.positions.append(new_pos)
                executed_syms.append(sym)
                
            for sym in executed_syms:
                del self.pending_orders[sym]

            # 1. Update Equity & Marginged) ...
            if current_day != current_time.date():
                current_day = current_time.date()
                daily_start_equity = self.account.equity
                trading_blocked_today = False
            
            open_pnl = 0.0
            for pos in self.account.positions:
                if pos.symbol in data:
                     try:
                        price = data[pos.symbol].at[current_time, 'Close']
                        gross = (price - pos.entry_price) * pos.size * pos.direction
                        open_pnl += gross
                     except (KeyError, ValueError):
                        pass
            
            self.account.equity = self.account.balance + open_pnl
            
            # ... (Daily Limit Check unchanged) ...
            daily_loss_pct = (self.account.equity - daily_start_equity) / daily_start_equity
            if not trading_blocked_today and daily_loss_pct < -(self.config.daily_loss_limit_pct / 100.0):
                 print(f"  [RISK] Daily Limit Hit ({daily_loss_pct:.2%}) at {current_time}. Closing All.")
                 trading_blocked_today = True
                 last_limit_hit_date = current_day
                 for pos in list(self.account.positions):
                     if pos.symbol in data and current_time in data[pos.symbol].index:
                         price = data[pos.symbol].at[current_time, 'Close']
                         self.account.close_position(pos, price, current_time, "DailyLimit")

            # === SIGNAL-DRIVEN EXIT LOGIC (MEDALLION STYLE) ===
            # Replaces Fixed SL/TP with Signal Decay/Flip checks + Emergency Stop
            
            for pos in list(self.account.positions): 
                if pos.symbol not in data: continue
                df = data[pos.symbol]
                if current_time not in df.index: continue
                
                row = df.loc[current_time]
                high = row['High']
                low = row['Low']
                close = row['Close']
                atr = row.get('ATR', 0.001)  # Fallback
                
                # Check Emergency SL first (Catastrophe Protection)
                # Calculate current PnL in ATR units
                current_pnl = (close - pos.entry_price) * pos.direction
                # Check Low/High for stop hit intrabar
                worst_price = low if pos.direction == 1 else high
                worst_pnl = (worst_price - pos.entry_price) * pos.direction
                
                # 1. Emergency SL (5x ATR)
                dist_atr = (pos.entry_price - worst_price) / atr if pos.direction == 1 else (worst_price - pos.entry_price) / atr
                
                # Check if price crossed emergency SL level
                emergency_price = pos.entry_price - (atr * self.config.emergency_sl_mult * pos.direction)
                emergency_hit = False
                
                if pos.direction == 1:
                     if low <= emergency_price: emergency_hit = True
                else: 
                     if high >= emergency_price: emergency_hit = True
                     
                if emergency_hit:
                    # Assume slippage or fill at stop
                    self.account.close_position(pos, emergency_price, current_time, "EmergencySL")
                    continue
                
                # 2. Key Signal Check
                # Retrieve current signal
                current_signal = row.get('Final_Signal', row.get('Ensemble_Score', 0))
                
                exit_reason = None
                
                # Signal Flip: Signal went against us significantly
                # (e.g. Long position, but Signal becomes <-0.1)
                if current_signal * pos.direction < -0.1:
                    exit_reason = "SignalFlip"
                
                # Signal Decay: Signal strength dropped below confidence threshold
                # (e.g. 0.8 -> 0.2)
                elif abs(current_signal) < self.config.signal_decay_threshold:
                    exit_reason = "SignalDecay"
                
                # Time Exit (Keep existing)
                elif (current_time - pos.entry_time).total_seconds() / 3600 > self.config.max_bars_in_trade:
                    exit_reason = "TimeExit"
                    
                if exit_reason:
                    self.account.close_position(pos, close, current_time, exit_reason)
            
            # 2. Check Entries
            if not trading_blocked_today:
                for sym, df in data.items():
                    if current_time not in df.index: continue
                    
                    # Check existing positions
                    current_positions = [p for p in self.account.positions if p.symbol == sym]
                    if len(current_positions) >= self.config.max_positions_per_symbol:
                        continue
                    if sym in self.pending_orders: # Don't stack orders
                        continue
                    if len(self.account.positions) >= self.config.max_concurrent_trades:
                        break
                    
                    row = df.loc[current_time]
                    score = row.get('Final_Signal', row.get('Ensemble_Score', 0))
                    close = row['Close']
                    high = row.get('High', close)
                    low = row.get('Low', close)
                    atr = row['ATR']
                    
                    # Update MFE/MAE for active positions
                    for pos in self.account.positions:
                        if pos.symbol == sym:
                            pos.update_extremes(high, low)
                    
                    direction = 0
                    if score > self.config.ensemble_threshold: direction = 1
                    elif score < -self.config.ensemble_threshold: direction = -1
                        
                    if direction != 0:
                        # Correlation Check (Simplified for brevity in diff)
                        # ... (Assume same logic as before) ...
                        
                        skip_due_to_correlation = False
                         # Correlation Helper (Compact)
                        def get_currencies(s):
                            if '=X' in s: return {s.replace('=X', '')[:3], s.replace('=X', '')[3:]}
                            return set()

                        if '=X' in sym:
                            sym_curr = get_currencies(sym)
                            for op in self.account.positions:
                                if '=X' in op.symbol and (sym_curr & get_currencies(op.symbol)):
                                    skip_due_to_correlation = True; break
                        if skip_due_to_correlation:
                            continue

                        # === MARGIN CHECK (Phase 9) ===
                        # 1. Calculate Current Notional
                        current_notional = 0.0
                        for p in self.account.positions:
                            # Approx Notional = Price * Size * 100000 (Units)
                            # We assume 'size' is in standard lots (100k)
                            current_notional += (p.entry_price * p.size * 100000)
                        
                        # 2. Calculate New Notional
                        # We need 'volume' first.
                        # Re-use Risk Logic briefly to estimate volume
                        balance = self.account.balance
                        if hasattr(self.account, 'equity'): balance = self.account.equity
                        
                        # [Duplicate of Live Logic - simplified]
                        # Vol Sizing
                        vol_int = row.get('Vol_Intensity', 0.0)
                        vol_mult = 1.0 / (1.0 + max(vol_int, 0.0)**2)
                        vol_mult = max(0.3, min(vol_mult, 1.2))
                        
                        risk_pct = self.config.risk_per_trade * vol_mult
                        risk_amt = balance * risk_pct
                        
                        sl_dist = atr * 1.5
                        # Approx value per lot (Standard Lot)
                        # For FX: 1 pip = $10. SL Dist (Price) / 0.0001 * 10
                        # Better: risk_amt / (sl_dist * 100000)
                        if sl_dist == 0: continue
                        
                        # Estimate Size (Lots)
                        # Assuming Quote Currency is USD-like (approx parity)
                        # This is an estimation for backtest
                        est_size = risk_amt / (sl_dist * 100000)
                        
                        new_notional = close * est_size * 100000
                        
                        total_notional = current_notional + new_notional
                        max_notional = balance * self.config.account_leverage
                        
                        # Rule A: Hard Margin (Can't exceed broker limit)
                        if total_notional > max_notional * 0.95: # 5% Buffer
                            # print(f"  [MARGIN BLOCK] {sym} Rejects. Notional {total_notional:,.0f} > Max {max_notional:,.0f}")
                            continue
                            
                        # Rule B: Soft Cap (6x Equity) - Prudent Risk
                        soft_cap = balance * 6.0
                        if total_notional > soft_cap:
                            # print(f"  [SOFT CAP BLOCK] {sym} Rejects. Notional {total_notional:,.0f} > Cap {soft_cap:,.0f}")
                            continue
                        
                        # --- OPTIMIZATION FILTERS (Phase 3) ---
                        # 1. Block "Toxic" Bull Trend Regime
                        trend_regime = row.get('Trend_Regime', 'RANGE')
                        if trend_regime == 'BULL':
                            continue
                            
                        # 2. Confidence Floor (High Confidence Only - Institutional)
                        prob_margin = row.get('prob_margin', 0.5)
                        if prob_margin < self.config.min_prob_margin:
                            continue
                        
                        # Entry Calculation
                        sl_dist = atr * self.config.emergency_sl_mult # For Risk Calc only
                        if direction == 1:
                            sl_price = close - sl_dist
                            tp_price = 0.0 # No fixed TP
                        else:
                            sl_price = close + sl_dist
                            tp_price = 0.0 # No fixed TP
                        
                        # Get Probability for Sizing
                        prob_win = row.get('prob_max', 0.5) if direction != 0 else 0.5
                        # If Short, prob_max is correct? prob_max is max(up, down). Yes.
                        
                        size = self.account.calculate_position_size(close, sl_dist, prob_win=prob_win)
                        
                        if size <= 0: continue
                        
                        entry_risk_dollars = size * sl_dist
                        
                        # 3. BLOCK High Volatility Entries (Council Fix)
                        # EXCEPTION: If High Confidence (Sniper), allow it.
                        vol_regime = row.get('Vol_Regime', 'NORMAL')
                        if vol_regime == 'HIGH':
                            if prob_margin < self.config.high_conf_threshold:
                                continue  # Block unless high confidence
                        
                        if last_limit_hit_date is not None:
                            if (current_day - last_limit_hit_date).days < self.config.drawdown_penalty_days:
                                size *= self.config.risk_penalty_factor
                        
                        # size = final_size # Removed redundant line
                        if size > 0:
                            # QUEUE ORDER FOR NEXT OPEN
                            # Avoid Lookahead Bias: Don't fill at Close.
                            
                            penalty_applied = False
                            if last_limit_hit_date is not None:
                                if (current_day - last_limit_hit_date).days < self.config.drawdown_penalty_days:
                                    penalty_applied = True
                            
                            risk_amt = self.account.equity * self.config.risk_per_trade
                            
                            context = {
                                'Risk_At_Entry': risk_amt,
                                'Entry_Trend_Regime': row.get('Trend_Regime', None),
                                'Entry_Vol_Regime': row.get('Vol_Regime', 'NORMAL'),
                                'Entry_Prob_Margin': row.get('prob_margin', 0.0),
                                'Signal_Score': score
                            }
                            
                            self.pending_orders[sym] = {
                                'direction': direction,
                                'score': score,
                                'sl_dist': sl_dist, # Store ATR-based dist
                                'prob_win': prob_win,
                                'penalty_applied': penalty_applied,
                                'context': context
                            }
            
            equity_curve.append(self.account.balance)
            
        # ... Reporting (Unchanged) ...
        final_balance = self.account.balance
        ret_pct = ((final_balance - self.config.initial_balance) / self.config.initial_balance) * 100
        
        print(f"  Simulation Complete.")
        print(f"  Final Balance: ${final_balance:,.2f}")
        print(f"  Total Return:  {ret_pct:.2f}%")
        print(f"  Trades Executed: {len(self.account.trade_history)}")

        win_trades = [t for t in self.account.trade_history if t['PnL'] > 0]
        if self.account.trade_history:
            win_rate = len(win_trades) / len(self.account.trade_history) * 100
            print(f"  Win Rate:      {win_rate:.1f}%")
        
        return pd.Series(equity_curve, index=combined_index)

# ============================================================================
# MONTE CARLO ENGINE (ROBUSTNESS)
# ============================================================================
class MonteCarloEngine:
    def __init__(self, config: Config):
        self.config = config
        
    def run_bootstrap_fractional(self, trades: List[Dict], n_sims: int = 1000, seed: int | None = None):
        """
        Monte Carlo using R-multiples and fractional equity sizing.
        Mirrors live behaviour and checks FTMO Daily/Total DD limits.
        """
        if not trades:
            print("[MC] No trades to simulate.")
            return

        print("\n[MONTE CARLO SKILLS CHECK] Fractional Sizing & FTMO Rules")
        print("-" * 40)
        
        if seed is not None:
            np.random.seed(seed)
            
        # Prepare Data
        R = np.array([t['R_Multiple'] for t in trades])
        dates = np.array([t['Entry Time'].date() for t in trades])
        
        initial_balance = self.config.initial_balance
        
        # FTMO Limits
        profit_target = initial_balance * (1 + self.config.mc_target_return / 100.0)
        overall_dd_limit = -self.config.overall_loss_limit_pct / 100.0
        daily_dd_limit = -self.config.daily_loss_limit_pct / 100.0
        
        print(f"  Target: ${profit_target:,.0f} (+{self.config.mc_target_return}%)")
        print(f"  Max DD: {self.config.overall_loss_limit_pct}% | Daily DD: {self.config.daily_loss_limit_pct}%")
        
        # Counters
        passed = 0
        failed_dd = 0
        failed_dd_daily = 0
        timeout = 0
        
        final_balances = []
        max_drawdowns_pct = []
        
        for s in range(n_sims):
            balance = initial_balance
            peak = balance
            overall_peak = balance
            
            # State
            current_day = None
            day_start_balance = balance
            
            hit_daily_limit = False
            hit_overall_limit = False
            hit_profit_target = False
            
            sim_max_dd = 0.0
            
            # Simulate Path (Resample N trades equal to history length)
            # Or should we simulate a fixed duration? "Bootstrap" usually implies N samples = N history.
            # Let's use len(R) trades per sim.
            
            indices = np.random.randint(0, len(R), size=len(R))
            
            for idx in indices:
                R_i = R[idx]
                day_i = dates[idx]
                
                # Day Change Logic
                if current_day != day_i:
                    current_day = day_i
                    day_start_balance = balance
                
                # Trade Outcome
                # Risk = % of Current Equity
                dollar_risk = balance * self.config.risk_per_trade
                pnl = R_i * dollar_risk
                balance += pnl
                
                # Update Peaks
                if balance > peak: peak = balance
                if balance > overall_peak: overall_peak = balance
                    
                # 1. Total Drawdown Check (Trailing from High Water Mark)
                dd_overall = (balance - overall_peak) / overall_peak
                if dd_overall < sim_max_dd: sim_max_dd = dd_overall
                
                if dd_overall <= overall_dd_limit:
                    hit_overall_limit = True
                    break
                    
                # 2. Daily Drawdown Check (From Day Start Equity)
                dd_daily = (balance - day_start_balance) / day_start_balance
                if dd_daily <= daily_dd_limit:
                    hit_daily_limit = True
                    hit_overall_limit = True # FTMO fails you for either
                    break
                    
                # 3. Profit Target
                if balance >= profit_target:
                    hit_profit_target = True
                    # In real challenge, you stop trading.
                    break
            
            final_balances.append(balance)
            max_drawdowns_pct.append(sim_max_dd)
            
            # Categorize
            if hit_profit_target and not hit_overall_limit:
                passed += 1
            elif hit_daily_limit:
                failed_dd_daily += 1
            elif hit_overall_limit: # Hit overall but not via daily
                failed_dd += 1
            else:
                timeout += 1 # Survived but didn't reach target in N trades
                
        # Statistics
        pass_rate = (passed / n_sims) * 100
        fail_daily_rate = (failed_dd_daily / n_sims) * 100
        fail_overall_rate = (failed_dd / n_sims) * 100
        timeout_rate = (timeout / n_sims) * 100
        
        median_bal = np.median(final_balances)
        var_95_bal = np.percentile(final_balances, 5)
        median_dd = np.median(max_drawdowns_pct)
        var_95_dd = np.percentile(max_drawdowns_pct, 5) # 5th percentile is the "bad tail" (e.g. -15%)
        
        print("\n  [OUTCOMES]")
        print(f"  PASS:              {pass_rate:.1f}%")
        print(f"  FAIL (Daily DD):   {fail_daily_rate:.1f}%")
        print(f"  FAIL (Total DD):   {fail_overall_rate:.1f}%")
        print(f"  TIMEOUT:           {timeout_rate:.1f}%")
        
        print(f"\n  [RISK PROFILE]")
        print(f"  Median Balance:    ${median_bal:,.2f}")
        print(f"  5% Tail Balance:   ${var_95_bal:,.2f}")
        print(f"  Median Max DD:     {median_dd:.2%}")
        print(f"  5% Tail Max DD:    {var_95_dd:.2%}")

    def run_bootstrap(self, trades: List[Dict], n_sims: int = 1000, seed: int = 42):
        if not trades:
            print("No trades to simulate.")
            return

        print(f"\n[MONTE CARLO SIMULATION]")
        print("-" * 40)
        
        pnls = np.array([t['PnL'] for t in trades])
        initial_balance = self.config.initial_balance
        
        final_balances = []
        max_drawdowns = []
        
        simulated_curves = []
        
        for i in range(n_sims):
            # 1. Resample PnLs with replacement
            sim_pnls = np.random.choice(pnls, size=len(pnls), replace=True)
            
            # 2. Construct Equity Curve
            sim_curve = np.concatenate([[initial_balance], initial_balance + np.cumsum(sim_pnls)])
            simulated_curves.append(sim_curve)
            
            final_balances.append(sim_curve[-1])
            
            # 3. Max Drawdown
            peak = np.maximum.accumulate(sim_curve)
            dd = (sim_curve - peak) / peak
            max_drawdowns.append(np.min(dd))
            
        # Analysis
        final_balances = np.array(final_balances)
        max_drawdowns = np.array(max_drawdowns)
        
        mean_ret = np.mean(final_balances)
        median_ret = np.median(final_balances)
        var_95 = np.percentile(final_balances, 5) # 5th percentile (Worst 5%)
        dd_95 = np.percentile(max_drawdowns, 5) # 5th percentile (Deepest DD)
        
        print(f"  Simulations: {n_sims}")
        print(f"  Median Balance: ${median_ret:,.2f}")
        print(f"  Mean Balance:   ${mean_ret:,.2f}")
        print(f"  VaR (95%):      ${var_95:,.2f}")
        print(f"  Max DD (95%):   {dd_95*100:.2f}%")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        # Plot first 50 curves
        for i in range(min(50, n_sims)):
            plt.plot(simulated_curves[i], color='gray', alpha=0.1)
            
        # Plot Median
        median_curve = np.median(np.array(simulated_curves), axis=0)
        plt.plot(median_curve, color='blue', label='Median Equity')
        
        plt.title(f"Monte Carlo Simulation ({n_sims} runs) - Trade Bootstrap")
        plt.xlabel("Trade Count")
        plt.ylabel("Equity ($)")
        plt.legend()
        plt.grid(True)
        out_path = "monte_carlo.png"
        plt.savefig(out_path)
        print(f"  Plot saved to {out_path}")
        
        # --- FTMO CHALLENGE PASS PROBABILITY ---
        print(f"\n[FTMO CHALLENGE ANALYSIS]")
        print("-" * 40)
        
        profit_target = initial_balance * 1.10  # 10% profit
        dd_limit = -0.10  # 10% drawdown
        
        passed = 0
        failed_dd = 0
        failed_timeout = 0
        
        for i in range(n_sims):
            curve = simulated_curves[i]
            peak = np.maximum.accumulate(curve)
            dd = (curve - peak) / peak
            
            hit_profit = curve >= profit_target
            hit_dd = dd <= dd_limit
            
            # Find first occurrence of each
            first_profit_idx = np.argmax(hit_profit) if np.any(hit_profit) else len(curve)
            first_dd_idx = np.argmax(hit_dd) if np.any(hit_dd) else len(curve)
            
            if np.any(hit_profit) and first_profit_idx < first_dd_idx:
                passed += 1
            elif np.any(hit_dd):
                failed_dd += 1
            else:
                failed_timeout += 1  # Neither hit (needs more trades)
        
        pass_rate = passed / n_sims * 100
        fail_dd_rate = failed_dd / n_sims * 100
        timeout_rate = failed_timeout / n_sims * 100
        
        print(f"  Profit Target: 10% (${profit_target:,.0f})")
        print(f"  DD Limit:      10%")
        print(f"  ---")
        print(f"  PASSED (10% before DD): {passed}/{n_sims} ({pass_rate:.1f}%)")
        print(f"  FAILED (DD first):      {failed_dd}/{n_sims} ({fail_dd_rate:.1f}%)")
        print(f"  PENDING (needs more):   {failed_timeout}/{n_sims} ({timeout_rate:.1f}%)")

    def run_crisis_stress_test(self, trades: List[dict], n_sims: int = 1000, initial_balance: float = 100000):
        """
        Simulate 1000 moderate-to-extreme crisis scenarios.
        Applies random shocks: volatility spikes, loss clustering, fat tails.
        """
        print(f"\n[CRISIS SCENARIO STRESS TEST]")
        print("-" * 40)
        print(f"  Simulating {n_sims} random crisis scenarios...")
        print(f"  Shock types: Volatility spikes (2-5x), Loss clustering, Fat tails")
        
        if len(trades) < 10:
            print("  ERROR: Not enough trades for stress test")
            return
        
        # Extract PnL values from trade dicts
        trade_pnls = np.array([t['PnL'] for t in trades])
        base_std = np.std(trade_pnls)
        
        results = {
            'final_balances': [],
            'max_dds': [],
            'survived': 0,
            'blew_up': 0,  # Hit 50%+ DD
            'ftmo_pass': 0,
            'ftmo_fail': 0,
            'severity': [],
        }
        
        for sim in range(n_sims):
            # Random crisis severity: 1.0 (normal) to 5.0 (extreme)
            severity = np.random.uniform(1.5, 5.0)
            results['severity'].append(severity)
            
            # Create stressed trade sequence
            stressed_pnls = trade_pnls.copy()
            
            # 1. Volatility spike: multiply losses by severity
            loss_mask = stressed_pnls < 0
            stressed_pnls[loss_mask] *= severity
            
            # 2. Loss clustering: with probability, cluster consecutive losses
            if np.random.random() < 0.5:  # 50% chance of clustering
                n_trades = len(stressed_pnls)
                cluster_start = np.random.randint(0, max(1, n_trades - 10))
                cluster_len = np.random.randint(3, min(10, n_trades - cluster_start))
                stressed_pnls[cluster_start:cluster_start + cluster_len] = -abs(stressed_pnls[cluster_start:cluster_start + cluster_len]) * severity
            
            # 3. Fat tail event: with probability, inject extreme loss
            if np.random.random() < 0.3:  # 30% chance of black swan
                extreme_loss = -base_std * np.random.uniform(5, 15) * severity
                inject_idx = np.random.randint(0, len(stressed_pnls))
                stressed_pnls[inject_idx] = extreme_loss
            
            # Bootstrap with stressed PnLs
            sampled = np.random.choice(stressed_pnls, size=len(stressed_pnls), replace=True)
            
            # Simulate equity curve
            balance = initial_balance
            peak = balance
            max_dd = 0
            
            for pnl in sampled:
                balance += pnl
                balance = max(balance, 0)  # Can't go negative
                
                if balance > peak:
                    peak = balance
                dd = (balance - peak) / peak if peak > 0 else 0
                if dd < max_dd:
                    max_dd = dd
            
            results['final_balances'].append(balance)
            results['max_dds'].append(max_dd)
            
            # Track outcomes
            if max_dd > -0.50:  # Didn't lose 50%+
                results['survived'] += 1
            else:
                results['blew_up'] += 1
            
            if balance >= initial_balance * 1.10 and max_dd > -0.10:
                results['ftmo_pass'] += 1
            elif max_dd <= -0.10:
                results['ftmo_fail'] += 1
        
        # Print results
        final_balances = np.array(results['final_balances'])
        max_dds = np.array(results['max_dds'])
        severity = np.array(results['severity'])
        
        print(f"\n  CRISIS STRESS TEST RESULTS:")
        print(f"  ---------------------------")
        print(f"  Crisis Severity Range:    {severity.min():.1f}x to {severity.max():.1f}x normal volatility")
        print(f"  Average Severity:         {severity.mean():.1f}x")
        print(f"")
        print(f"  Median Final Balance:     ${np.median(final_balances):,.2f}")
        print(f"  Mean Final Balance:       ${np.mean(final_balances):,.2f}")
        print(f"  Worst Case Balance:       ${np.min(final_balances):,.2f}")
        print(f"  VaR (95%):                ${np.percentile(final_balances, 5):,.2f}")
        print(f"")
        print(f"  Max DD (Median):          {np.median(max_dds)*100:.2f}%")
        print(f"  Max DD (95th percentile): {np.percentile(max_dds, 5)*100:.2f}%")
        print(f"  Max DD (Worst):           {np.min(max_dds)*100:.2f}%")
        print(f"")
        print(f"  SURVIVAL ANALYSIS:")
        print(f"  -----------------")
        print(f"  Survived (<50% DD):       {results['survived']}/{n_sims} ({results['survived']/n_sims*100:.1f}%)")
        print(f"  Blew Up (>50% DD):        {results['blew_up']}/{n_sims} ({results['blew_up']/n_sims*100:.1f}%)")
        print(f"")
        print(f"  FTMO Under Crisis:")
        print(f"  ------------------")
        print(f"  Would PASS FTMO:          {results['ftmo_pass']}/{n_sims} ({results['ftmo_pass']/n_sims*100:.1f}%)")
        print(f"  Would FAIL FTMO:          {results['ftmo_fail']}/{n_sims} ({results['ftmo_fail']/n_sims*100:.1f}%)")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    config = Config()
    
    # Calculate Rolling Window (Max ~730 days for 1h yfinance)
    end_date = datetime.now()
    # Default config.ml_lookback_days = 729
    start_date = end_date - timedelta(days=config.ml_lookback_days)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"\n============================================================")
    print(f"ROLLING WINDOW SYSTEM (Max History: {config.ml_lookback_days} days)")
    print(f"Range: {start_str} -> {end_str}")
    print(f"Split: {config.ml_train_split_pct*100:.0f}% Train / {(1-config.ml_train_split_pct)*100:.0f}% Test")
    print(f"============================================================\n")

    loader = DataLoader(config)
    fe = FeatureEngine(config)
    re = RegimeEngine(config)
    alpha = AlphaEngine(config)
    ens = EnsembleSignal(config)
    crisis = CrisisAlphaEngine(config)
    
    try:
        # 1. Load All Data (Rolling Window)
        print("[STEP 1] Loading Data...")
        data = loader.load_data(start_str, end_str)

        # 2. Features & Regimes
        print("[STEP 2] Computing Features...")
        data = fe.add_features_all(data)
        data = re.add_regimes_all(data)

        # 3. Walk-Forward Backtest (To generate 1.5 years of OOS stats)
        # We have ~730 days of data.
        # Strategy:
        # Start with 365 days training.
        # Test 1 month (or 1 week).
        # Expand window.
        
        print("\n[STEP 3-6] Running Walk-Forward Optimization (Long Diagnostic Run)...")
        print("-" * 40)
        
        full_df_map = data
        # Align indices
        combined_idx = pd.Index([])
        for df in full_df_map.values():
            combined_idx = combined_idx.union(df.index)
        combined_idx = combined_idx.sort_values()
        
        start_date = combined_idx[0]
        end_date = combined_idx[-1]
        
        # Define Walk-Forward Parameters
        train_window_days = 180 # Short training window to adapt fast
        test_window_days = 30
        
        current_date = start_date + pd.Timedelta(days=train_window_days)
        
        oos_trades = []
        oos_equity = [config.initial_balance]
        
        # Initialize Backtester to track state persists across windows?
        # Actually, best to just aggregate trade lists.
        # We need a Persistent Backtester.
        
        pb = Backtester(config)
        
        while current_date < end_date:
            train_start = start_date # Expanding window or Rolling? Rolling is better for regime adaptation.
            # train_start = current_date - pd.Timedelta(days=train_window_days) 
            # Let's use Expanding for stability initially, or Rolling?
            # User said "Freshness is Key". Rolling 6 months seems good.
            train_start = current_date - pd.Timedelta(days=train_window_days)
            if train_start < start_date: train_start = start_date
            
            test_end = current_date + pd.Timedelta(days=test_window_days)
            if test_end > end_date: test_end = end_date
            
            print(f"  > Window: Train {train_start.date()}->{current_date.date()} | Test ->{test_end.date()}")
            
            # Slice Data
            train_data = {}
            test_data = {}
            for sym, df in full_df_map.items():
                train_data[sym] = df.loc[train_start:current_date].copy()
                test_data[sym] = df.loc[current_date:test_end].copy()
            
            # Train Alpha
            alpha.train_model(train_data)
            
            # Generate Signals on Test
            test_data_sig = alpha.add_signals_all(test_data)
            test_data_ens = ens.add_ensemble_all(test_data_sig)
            test_data_final = crisis.add_crisis_signals(test_data_ens)
            
            # Run Backtest on this chunk
            # We need to feed the Persistent Backtester only this chunk
            # No, `Backtester.__init__` creates new `Account`.
            
            # Let's write a targeted `run_walk_forward_chunk` on the fly or just use a loop here.
            # Given constraints, I will rely on `Backtester` logic but manually patch equity.
            
            chunk_bt = Backtester(config)
            chunk_bt.account.balance = pb.account.balance
            chunk_bt.account.equity = pb.account.equity
            chunk_bt.account.peak_equity = pb.account.peak_equity
            
            # Copy positions?
            # If we cross boundaries with open positions, this is tricky.
            # Simplified Walk Forward: Valid for "Diagnostic" -> Close all at end of month or ignore overlap.
            # Better: Carry over positions.
            chunk_bt.account.positions = pb.account.positions
            
            chunk_bt.run_backtest(test_data_final)
            
            # Update Persistent State
            pb.account.balance = chunk_bt.account.balance
            pb.account.equity = chunk_bt.account.equity
            pb.account.peak_equity = chunk_bt.account.peak_equity
            pb.account.positions = chunk_bt.account.positions
            pb.account.trade_history.extend(chunk_bt.account.trade_history)
            
            current_date = test_end
            
        print("\n[WALK-FORWARD COMPLETE]")
        print(f"  Final Balance: ${pb.account.balance:,.2f}")
        print(f"  Total Trades: {len(pb.account.trade_history)}")
        
        # Use the aggregated history for Diagnostics
        bt = pb # Assign to bt so downstream calls work
        
        # [STEP 7] Running Verification & Monte Carlo
        print("\n[STEP 7] Running Verification & Monte Carlo...")
        
        # 7.1 Diagnostic Analysis (Phase 2)
        analyze_trades(bt.account.trade_history)

        # 7.2 Monte Carlo Simulations
        monte_carlo = MonteCarloEngine(config)
        
        print("\n[MC MODE 1] Fixed-Dollar Bootstrap (Conservative)")
        monte_carlo.run_bootstrap(bt.account.trade_history, n_sims=config.mc_simulations)
        
        print("\n[MC MODE 2] Fractional R-Multiple Bootstrap (FTMO-Aware)")
        monte_carlo.run_bootstrap_fractional(bt.account.trade_history, n_sims=config.mc_simulations)
        
        if bt.account.trade_history:
            df_trades = pd.DataFrame(bt.account.trade_history)
            df_trades.to_csv('backtest_results.csv', index=False)
            print(f"\n[SAVED] Trade log saved to backtest_results.csv ({len(df_trades)} trades)")
            
        print("\n[FTMO CHALLENGE ANALYSIS FINALIZED]") 
        
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("Verification Complete.")

if __name__ == "__main__":
    main()
