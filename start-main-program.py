import os
import time
import subprocess
from datetime import datetime

def get_next_month_first():
    """Calculate the datetime for the first day of the next month at 00:00:00."""
    now = datetime.now()
    year = now.year
    month = now.month + 1
    if month > 12:
        month = 1
        year += 1
    return datetime(year, month, 1, 0, 0, 0)

def determine_script(month):
    """Determine which script to run based on the month."""
    if month not in [5, 6, 7, 11]:
        return "billionaire-strategy-buy-lowest-price-stock-market-robot.py"
    else:
        return "stock-market-robot.py"

# Initial setup: delete db and start the appropriate robot
os.system("rm trading_bot.db")
current_month = datetime.now().month
current_script = determine_script(current_month)
current_process = subprocess.Popen(["python3", current_script])
print(f"Started {current_script} (Month: {current_month})")

# Start the two continuous scripts in separate gnome-terminal windows
stock_list_process = subprocess.Popen(
    ["gnome-terminal", "--", "python3", "stock-list-writer-for-list-of-stock-symbols-to-scan.py"]
)
performance_process = subprocess.Popen(
    ["gnome-terminal", "--", "python3", "performance-stock-list-writer.py"]
)
print("Started stock-list-writer-for-list-of-stock-symbols-to-scan.py in new terminal")
print("Started performance-stock-list-writer.py in new terminal")

# Main loop to check on the first of each next month
while True:
    next_first = get_next_month_first()
    sleep_seconds = (next_first - datetime.now()).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    
    # Now it's approximately the 1st of the next month
    now = datetime.now()
    new_month = now.month
    new_script = determine_script(new_month)
    
    if new_script != current_script:
        # Kill the previous process
        current_process.terminate()
        current_process.wait()
        
        # Delete the database
        os.system("rm trading_bot.db")
        
        # Start the new script
        current_process = subprocess.Popen(["python3", new_script])
        current_script = new_script
        print(f"Switched to {new_script} (Month: {new_month})")
    else:
        print(f"No switch needed, continuing with {current_script} (Month: {new_month})")
