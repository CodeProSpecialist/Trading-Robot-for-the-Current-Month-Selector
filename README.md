# Trading-Robot-for-the-Current-Month-Selector
This is a combination of the Billionaire Trading Robot and the Bull Market Robot and it picks the best robot for the current Earnings Season. 

***** Upgrade to the newest version of this Python Robot today because some Python code updates were finished and some errors were recently fixed on September 16, 2025. *****

Run the following command in a command line window terminal:  

cd ~/Trading-Robot-for-the-Current-Month-Selector

On the first run, install python dependencies: 

Install Anaconda Python 3.13.5 
with the default 
environment called "base."

conda activate

bash install.sh

bash install_dependencies.sh

( run the following python 3 script )

python3 start-main-program.py

( Recommended Operating System: 
Ubuntu 24.04 LTS Linux )

How does the Billionaire Strategy Stock Market Trading Program Work? 

The stock market trading robot automates buying and selling decisions for S&P 500 stocks listed in electricity-or-utility-stocks-to-buy-list.txt, which is populated by a separate script filtering S&P 500 stocks based on performance and technical criteria. Despite the file name, the robot trades across the broad S&P 500 market, leveraging a robust stock selection and trading strategy to maximize profitability. Below are the key features driving its potential profitability.

1. Advanced Stock Selection





Performance-Based Filtering: The selection script filters S&P 500 stocks (e.g., AAPL, MSFT, NVDA) requiring positive percentage changes over 30 days and 5 days, ensuring short-term and medium-term momentum.



Comprehensive Technical Scoring: Stocks are scored using multiple indicators:





RSI (14-period, neutral 30–70 or oversold ≤ 30), MACD (12, 26, 9), VWAP (14-period), Bollinger Bands (20-period), Stochastic Oscillator (14, 3), ADX (14-period, > 25 for strong trends), and OBV (accumulation).



Price increases ≥ 5% over 1- or 2-year lookbacks, seasonal returns > 5% for the current month, and bonuses for stocks in their historically best-performing month.



Sector Diversification: Limits selections to the top 5 stocks per sector, then picks the top 30 overall, ensuring a diversified S&P 500 portfolio written to electricity-or-utility-stocks-to-buy-list.txt.



Efficient Processing: Uses parallel processing (up to 20 threads) to analyze hundreds of S&P 500 stocks quickly, minimizing delays in stock selection.

2. Technical Trading Strategy





Buy Signals: The robot buys when RSI ≥ 65 (strong momentum) or the current price is 0.2% below the recent price (0.998 × last price), targeting dips in S&P 500 stocks.



Trend and Volatility Adaptation: Uses MACD to confirm trends and ATR to set dynamic buy (current price - 0.10 × ATR) and sell (current price + 0.40 × ATR) signals, adapting to S&P 500 stock volatility.



Profit-Taking Discipline: Sells stocks held for at least one day when the price exceeds 0.5% above the buy price, locking in small, consistent gains.

3. Automated Execution





Fractional Shares: Executes market orders via Alpaca’s API with notional values (up to $600 per stock), allowing precise capital allocation across S&P 500 stocks.



Trailing Stops: Places 1% trailing stop-loss orders for whole-share quantities to secure profits or limit losses, though fractional shares lack this protection due to API constraints.

4. Risk Management





Cash Allocation: Distributes cash equally (max $600 per stock) while maintaining a $1.00 minimum balance, preventing overexposure in the S&P 500.



Day Trade Compliance: Limits day trades to 3 in 5 business days, ensuring regulatory compliance and continuous trading.



Error Resilience: Handles API or data errors with try-except blocks, logging issues, and pausing for 120 seconds (trading) or 300 seconds (stock selection) to recover, ensuring operational stability.

5. Real-Time Monitoring





Price Updates: Retrieves real-time prices from a market data source during pre-market, regular, and post-market hours, with fallbacks to last closing prices, ensuring reliable decisions for S&P 500 stocks.



Market Hours Focus: Operates only during market hours (9:30 AM–4:00 PM Eastern, Monday–Friday), avoiding low-liquidity periods.

6. Data Persistence and Logging





SQLite Database: Tracks trade history and positions in trading_bot.db using SQLAlchemy, ensuring accurate multi-day position management for S&P 500 stocks.



CSV Logging: Logs trades in log-file-of-buy-and-sell-signals.csv and stock selection in stock_scanner.log, enabling performance analysis and strategy refinement.



Thread Safety: Uses buy_sell_lock to prevent race conditions during concurrent buy/sell operations.

7. Execution Efficiency





Multithreading: Employs separate threads for buying and selling, enabling simultaneous trade execution to capture S&P 500 market opportunities.



Batch Data Retrieval: Stock selection uses batch downloads with fallback to smaller batches, reducing API rate-limit risks.

8. Transparency and Debugging





Configurable Outputs: Flags (PRINT_STOCKS_TO_BUY, PRINT_DATABASE, DEBUG) display stock lists, technical indicators, and database contents for monitoring S&P 500 stock performance.



Stock Selection Table: Outputs a table of top stocks with metrics (score, RSI, volume ratio, etc.), enhancing transparency.



Financial Oversight: Displays cash balance and day trade counts for user awareness.

Profitability Drivers





High-Potential Stocks: The rigorous selection process identifies S&P 500 stocks with strong momentum, technical signals, and seasonal performance, increasing profitable trade likelihood.



Market Adaptability: RSI, MACD, and ATR in trading, combined with diverse indicators in selection, capture S&P 500 opportunities across sectors.



Risk Mitigation: Trailing stops, cash limits, and compliance checks reduce losses in the volatile S&P 500 market.



Automation: Multithreading and real-time data enable rapid, disciplined execution, critical for S&P 500 market movements.



Reliability: Error handling, retries, and data persistence ensure continuous operation and accurate position tracking.

Important: Don't forget to regularly update your list of stocks to buy and keep an eye on the market conditions. Happy trading!

Remember that all trading involves risks. The ability to successfully implement these strategies depends on both market conditions and individual skills and knowledge. As such, trading should only be done with funds that you can afford to lose. Always do thorough research before making investment decisions, and consider consulting with a financial advisor. This is use at your own risk software. This software does not include any warranty or guarantees other than the useful tasks that may or may not work as intended for the software application end user. The software developer shall not be held liable for any financial losses or damages that occur as a result of using this software for any reason to the fullest extent of the law. Using this software is your agreement to these terms. This software is designed to be helpful and useful to the end user.

Place your alpaca code keys in the location: /home/name-of-your-home-folder/.bashrc Be careful to not delete the entire .bashrc file. Just add the 4 lines to the bottom of the .bashrc text file in your home folder, then save the file. .bashrc is a hidden folder because it has the dot ( . ) in front of the name. Remember that the " # " pound character will make that line unavailable. To be helpful, I will comment out the real money account for someone to begin with an account that does not risk using real money. The URL with the word "paper" does not use real money. The other URL uses real money. Making changes here requires you to reboot your computer or logout and login to apply the changes.

The 4 lines to add to the bottom of .bashrc are:

export APCA_API_KEY_ID='zxzxzxzxzxzxzxzxzxzxz'

export APCA_API_SECRET_KEY='zxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzxzx'

#export APCA_API_BASE_URL='https://api.alpaca.markets'

export APCA_API_BASE_URL='https://paper-api.alpaca.markets'
