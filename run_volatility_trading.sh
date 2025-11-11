#!/bin/bash
# Run volatility trading algorithm in tmux with caffeinate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMUX_SESSION="volatility_trading"

# Check if tmux session already exists
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Session '$TMUX_SESSION' already exists. Attaching..."
    tmux attach-session -t "$TMUX_SESSION"
else
    echo "Starting new tmux session '$TMUX_SESSION'..."
    # Create new tmux session and run the algorithm with caffeinate
    tmux new-session -d -s "$TMUX_SESSION" -c "$SCRIPT_DIR" \
        "caffeinate -i uv run python personal/volatility_trading.py"
    
    echo "Session started. Attaching..."
    tmux attach-session -t "$TMUX_SESSION"
fi

