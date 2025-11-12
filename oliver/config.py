"""
Configuration constants for volatility trading algorithm.
Centralizes all tunable parameters and settings.
"""

import os
import pytz

# Trading parameters
BASE_SPREAD_CENTS = 7
VOLATILITY_MULTIPLIER = 10000
MAX_SPREAD_CENTS = 50
VOLUME = 25
USE_KELLY_SIZING = False
MAX_POSITION_PCT_OF_BALANCE = 0.20
KELLY_FRACTION = 0.25
MIN_POSITION_SIZE = 5
CYCLE_INTERVAL_SECONDS = 1
MARKET_MAKING_BUFFER_CENTS = 1
MARKET_TAKING_FEE_RATE = 0.07

# Data frequency settings
DATA_STEP_SECONDS = 60
DATA_HOURS_BACK = 6

# Prediction settings
PREDICTION_MODEL = 'gbm'
PREDICTION_EXPONENTIAL_DECAY = 0.5

# Spike detection and trading pause
SPIKE_DETECTION_ZSCORE_THRESHOLD = 3.0
SPIKE_RESUME_ZSCORE_THRESHOLD = 1.5
SPIKE_MIN_DATA_POINTS = 30
SPIKE_PAUSE_DURATION_SECONDS = 30
SPIKE_CUMULATIVE_DEVIATION_THRESHOLD = 250.0
SPIKE_CUMULATIVE_RESUME_THRESHOLD = 50.0

# Balance floor protection
BALANCE_FLOOR_BUFFER = 50.0

# Log directory
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Timezone
EST = pytz.timezone('US/Eastern')

