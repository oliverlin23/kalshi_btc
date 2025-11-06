#!/usr/bin/env python3
"""
Simple runner script for the BTC dashboard.
Run this from the project root.
"""

import sys
import os

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Run the dashboard
from app.btc_dashboard import main

if __name__ == "__main__":
    main()

