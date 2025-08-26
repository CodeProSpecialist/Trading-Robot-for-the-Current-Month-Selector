import os
import time
import subprocess
from datetime import datetime
from pathlib import Path
import sys

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

# Paths
anaconda_path = Path.home() / "anaconda3"
conda_sh_path = anaconda_path / "etc" / "profile.d" / "conda.sh"
project_dir = Path.home() / "Trading-Robot-for-the-Current-Month-Selector"

# Step 1: Check for Anaconda3 and project folder
if not anaconda_path.exists() or not conda_sh_path.exists():
    print("\n⚠️  Anaconda3 not found at ~/anaconda3.")
    print("Please download and install Anaconda from:\n👉 https://www.anaconda.com/download\n")
    sys.exit(1)

if not project_dir.exists():
    print("\n⚠️  Project folder not found at ~/Trading-Robot-for-the-Current-Month-Selector.")
    print("Please make sure the folder exists and contains your scripts.")
    sys.exit(1)

# Step 2: Change working directory to the project folder
os.chdir(project_dir)
print(f"Working directory changed to: {project_dir}")

# Step 3: Initial setup
os.system("rm -f trading_bot.db")
current_month = datetime.now().month
current_script = determine_script(current_month)

# Launch the main script in the current terminal with conda activated
current_process = subprocess.Popen([
    "bash", "-c",
    f"source {conda_sh_path} && conda activate base && cd {project_dir} && python3 {current_script}"
])
print(f"Started {current_script} (Month: {current_month})")

# Step 4: Start the two continuous scripts in new terminal windows
stock_list_process = subprocess.Popen([
    "x-terminal-emulator", "-e",
    f"bash -c 'source {conda_sh_path} && conda activate base && cd {project_dir} && python3 stock-list-writer-for-list-of-stock-symbols-to-scan.py; exec bash'"
])

performance_process = subprocess.Popen([
    "x-terminal-emulator", "-e",
    f"bash -c 'source {conda_sh_path} && conda activate base && cd {project_dir} && python3 auto-copy-stock-list-writer.py; exec bash'"
])

print("Started stock-list-writer-for-list-of-stock-symbols-to-scan.py in new terminal")
print("Started auto-copy-stock-list-writer.py in new terminal")

# Step 5: Monthly check loop
while True:
    next_first = get_next_month_first()
    sleep_seconds = (next_first - datetime.now()).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    now = datetime.now()
    new_month = now.month
    new_script = determine_script(new_month)

    if new_script != current_script:
        current_process.terminate()
        current_process.wait()
        os.system("rm -f trading_bot.db")
        current_process = subprocess.Popen([
            "bash", "-c",
            f"source {conda_sh_path} && conda activate base && cd {project_dir} && python3 {new_script}"
        ])
        current_script = new_script
        print(f"Switched to {new_script} (Month: {new_month})")
    else:
        print(f"No switch needed, continuing with {current_script} (Month: {new_month})")
