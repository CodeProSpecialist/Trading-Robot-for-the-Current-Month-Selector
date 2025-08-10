import os
from datetime import datetime

def get_current_month():
    """Returns the current month as a number (1-12)."""
    return datetime.now().month

def run_trading_robot():
    """Determines which trading robot script to run based on the current month."""
    current_month = get_current_month()
    
    # Months that are NOT May(5), June(6), July(7), or November(11)
    if current_month not in [5, 6, 7, 11]:
        os.system("python3 billionaire-strategy-buy-lowest-price-stock-market-robot.py")
        print(f"Running billionaire-strategy-buy-lowest-price-stock-market-robot.py (Month: {current_month})")
    else:
        os.system("python3 stock-market-robot.py")
        print(f"Running stock-market-robot.py (Month: {current_month})")

# Run the trading robot once and exit
run_trading_robot()
