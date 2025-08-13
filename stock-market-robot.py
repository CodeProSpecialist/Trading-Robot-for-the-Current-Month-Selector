import threading
import logging
import csv
import os
import time
from datetime import datetime, timedelta, date
from datetime import time as time2
import alpaca_trade_api as tradeapi
import pytz
import talib
import yfinance as yf
import sqlalchemy
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables for Alpaca API
APIKEYID = os.getenv('APCA_API_KEY_ID')
APISECRETKEY = os.getenv('APCA_API_SECRET_KEY')
APIBASEURL = os.getenv('APCA_API_BASE_URL')

# Initialize the Alpaca API
api = tradeapi.REST(APIKEYID, APISECRETKEY, APIBASEURL)

global stocks_to_buy, today_date, today_datetime, csv_writer, csv_filename, fieldnames, price_changes, end_time

# Configuration flags
PRINT_STOCKS_TO_BUY = False  # Keep as False for faster execution
PRINT_ROBOT_STORED_BUY_AND_SELL_LIST_DATABASE = True  # Keep as True to view database
PRINT_DATABASE = True  # Keep as True to view stocks to sell
DEBUG = False  # Keep as False for faster execution

# Set the timezone to Eastern
eastern = pytz.timezone('US/Eastern')

# Dictionary to maintain previous prices and price increase and decrease counts
stock_data = {}
previous_prices = {}
price_changes = {}
end_time = 0  # Initialize end_time as a global variable

# Define the API datetime format
api_time_format = '%Y-%m-%dT%H:%M:%S.%f-04:00'

# Thread lock for thread-safe operations
buy_sell_lock = threading.Lock()

# Logging configuration
logging.basicConfig(filename='trading-bot-program-logging-messages.txt', level=logging.INFO)

# Define the CSV file and fieldnames
csv_filename = 'log-file-of-buy-and-sell-signals.csv'
fieldnames = ['Date', 'Buy', 'Sell', 'Quantity', 'Symbol', 'Price Per Share']

# Initialize CSV file
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# Define the Database Models
Base = sqlalchemy.orm.declarative_base()

class TradeHistory(Base):
    __tablename__ = 'trade_history'
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    action = Column(String)  # 'buy' or 'sell'
    quantity = Column(Float)  # Changed to Float for fractional shares
    price = Column(Float)
    date = Column(String)

class Position(Base):
    __tablename__ = 'positions'
    symbol = Column(String, primary_key=True)
    quantity = Column(Float)  # Changed to Float for fractional shares
    avg_price = Column(Float)
    purchase_date = Column(String)

# Initialize SQLAlchemy
engine = create_engine('sqlite:///trading_bot.db')
Session = sessionmaker(bind=engine)
session = Session()

# Create tables if not exist
Base.metadata.create_all(engine)

def stop_if_stock_market_is_closed():
    market_open_time = time2(9, 27)
    market_close_time = time2(16, 0)
    while True:
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()
        if now.weekday() <= 4 and market_open_time <= current_time <= market_close_time:
            break
        print("\n")
        print('''
            2025 Edition of the Bull Market Advanced Stock Market Trading Robot, Version 8 
            
                           https://github.com/CodeProSpecialist
                       Featuring an Accelerated Database Engine with Python 3 SQLAlchemy  
         ''')
        print(f'Current date & time (Eastern Time): {now.strftime("%A, %B %d, %Y, %I:%M:%S %p")}')
        print("Stockbot only works Monday through Friday: 9:30 am - 4:00 pm Eastern Time.")
        print("Stockbot begins watching stock prices early at 9:27 am Eastern Time.")
        print("Waiting until Stock Market Hours to begin the Stockbot Trading Program.")
        print("\n")
        time.sleep(60)

def print_database_tables():
    if PRINT_DATABASE:
        positions = api.list_positions()
        show_price_percentage_change = True
        print("\nTrade History In This Robot's Database:")
        print("\n")
        print("Stock | Buy or Sell | Quantity | Avg. Price | Date ")
        print("\n")
        for record in session.query(TradeHistory).all():
            print(f"{record.symbol} | {record.action} | {record.quantity:.4f} | {record.price:.2f} | {record.date}")
        print("----------------------------------------------------------------")
        print("\n")
        print("Positions in the Database To Sell 1 or More Days After the Date Shown:")
        print("\n")
        print("Stock | Quantity | Avg. Price | Date or The 1st Day This Robot Began Working ")
        print("\n")
        for record in session.query(Position).all():
            symbol, quantity, avg_price, purchase_date = record.symbol, record.quantity, record.avg_price, record.purchase_date
            purchase_date_str = purchase_date
            if show_price_percentage_change:
                current_price = get_current_price(symbol)
                percentage_change = ((current_price - avg_price) / avg_price) * 100 if current_price and avg_price else 0
                print(f"{symbol} | {quantity:.4f} | {avg_price:.2f} | {purchase_date_str} | Price Change: {percentage_change:.2f}%")
            else:
                print(f"{symbol} | {quantity:.4f} | {avg_price:.2f} | {purchase_date_str}")
        print("\n")

def get_stocks_to_trade():
    try:
        with open('electricity-or-utility-stocks-to-buy-list.txt', 'r') as file:
            stock_symbols = [line.strip() for line in file.readlines()]
        if not stock_symbols:
            print("\n")
            print("********************************************************************************************************")
            print("*   Error: The file electricity-or-utility-stocks-to-buy-list.txt doesn't contain any stock symbols.   *")
            print("*   This Robot does not work until you place stock symbols in the file named:                          *")
            print("*       electricity-or-utility-stocks-to-buy-list.txt                                                  *")
            print("********************************************************************************************************")
            print("\n")
        return stock_symbols
    except FileNotFoundError:
        print("\n")
        print("****************************************************************************")
        print("*   Error: File not found: electricity-or-utility-stocks-to-buy-list.txt   *")
        print("****************************************************************************")
        print("\n")
        return []

def remove_symbol_from_trade_list(symbol):
    with open('electricity-or-utility-stocks-to-buy-list.txt', 'r') as file:
        lines = file.readlines()
    with open('electricity-or-utility-stocks-to-buy-list.txt', 'w') as file:
        for line in lines:
            if line.strip() != symbol:
                file.write(line)

def get_opening_price(symbol):
    symbol = symbol.replace('.', '-')
    stock_data = yf.Ticker(symbol)
    try:
        opening_price = round(float(stock_data.history(period="1d")["Open"].iloc[0]), 4)
        return opening_price
    except IndexError:
        logging.error(f"Opening price not found for {symbol}.")
        return None

def get_current_price(symbol):
    symbol = symbol.replace('.', '-')
    stock_data = yf.Ticker(symbol)
    try:
        current_price = round(float(stock_data.history(period='1d')['Close'].iloc[-1]), 4)
        return current_price
    except IndexError:
        logging.error(f"Current price not found for {symbol}.")
        return None

def get_atr_high_price(symbol):
    atr_value = get_average_true_range(symbol)
    current_price = get_current_price(symbol)
    return round(current_price + 0.40 * atr_value, 4) if current_price and atr_value else None

def get_atr_low_price(symbol):
    atr_value = get_average_true_range(symbol)
    current_price = get_current_price(symbol)
    return round(current_price - 0.10 * atr_value, 4) if current_price and atr_value else None

def get_average_true_range(symbol):
    symbol = symbol.replace('.', '-')
    ticker = yf.Ticker(symbol)
    data = ticker.history(period='30d')
    try:
        atr = talib.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=22)
        return atr[-1]
    except Exception as e:
        logging.error(f"Error calculating ATR for {symbol}: {e}")
        return None

def status_printer_buy_stocks():
    print(f"\rBuy stocks function is working correctly right now. Checking stocks to buy.....", end='', flush=True)
    print()

def status_printer_sell_stocks():
    print(f"\rSell stocks function is working correctly right now. Checking stocks to sell.....", end='', flush=True)
    print()

def calculate_technical_indicators(symbol, lookback_days=90):
    symbol = symbol.replace('.', '-')
    stock_data = yf.Ticker(symbol)
    historical_data = stock_data.history(period=f'{lookback_days}d')
    short_window = 12
    long_window = 26
    signal_window = 9
    historical_data['macd'], historical_data['signal'], _ = talib.MACD(historical_data['Close'],
                                                                       fastperiod=short_window,
                                                                       slowperiod=long_window,
                                                                       signalperiod=signal_window)
    rsi_period = 14
    historical_data['rsi'] = talib.RSI(historical_data['Close'], timeperiod=rsi_period)
    historical_data['volume'] = historical_data['Volume']
    return historical_data

def print_technical_indicators(symbol, historical_data):
    print("")
    print(f"\nTechnical Indicators for {symbol}:\n")
    print(historical_data[['Close', 'macd', 'signal', 'rsi', 'volume']].tail())
    print("")

def calculate_cash_on_hand():
    cash_available = round(float(api.get_account().cash), 2)
    return cash_available

def calculate_total_symbols(stocks_to_buy):
    total_symbols = len(stocks_to_buy)
    return total_symbols

def allocate_cash_equally(cash_available, total_symbols):
    max_allocation_per_symbol = 600.0  # Maximum dollar amount per stock
    allocation_per_symbol = min(max_allocation_per_symbol, cash_available / total_symbols) if total_symbols > 0 else 0
    return round(allocation_per_symbol, 2)

def get_previous_price(symbol):
    if symbol in previous_prices:
        return previous_prices[symbol]
    else:
        current_price = get_current_price(symbol)
        previous_prices[symbol] = current_price
        print(f"No previous price for {symbol} was found. Using the current price as the previous price: {current_price}")
        return current_price

def update_previous_price(symbol, current_price):
    previous_prices[symbol] = current_price

def track_price_changes(symbol, duration=180):
    global end_time
    start_time = time.time()
    end_time = start_time + duration
    while time.time() < end_time:
        try:
            current_price = get_current_price(symbol)
            with buy_sell_lock:  # Ensure thread-safe access to shared data
                previous_price = get_previous_price(symbol)
                print("")
                print_technical_indicators(symbol, calculate_technical_indicators(symbol))
                print("")
                if symbol not in price_changes:
                    price_changes[symbol] = {'increased': 0, 'decreased': 0}
                if current_price > previous_price:
                    price_changes[symbol]['increased'] += 1
                    print(f"{symbol} price just increased | current price: {current_price}")
                elif current_price < previous_price:
                    price_changes[symbol]['decreased'] += 1
                    print(f"{symbol} price just decreased | current price: {current_price}")
                else:
                    print(f"{symbol} price has not changed | current price: {current_price}")
                update_previous_price(symbol, current_price)
        except Exception as e:
            print(f"Error monitoring {symbol}: {str(e)}")
            logging.error(f"Error monitoring {symbol}: {str(e)}")
        time.sleep(10)  # Check price every 10 seconds
    print(f"Completed 3-minute monitoring for {symbol}.")

def end_time_reached():
    return time.time() >= end_time

def buy_stocks(bought_stocks, stocks_to_buy, buy_sell_lock):
    stocks_to_remove = []
    global start_time, end_time, original_start_time, price_changes, buy_stock_green_light
    buy_stock_green_light = 0
    extracted_date_from_today_date = datetime.today().date()
    today_date_str = extracted_date_from_today_date.strftime("%Y-%m-%d")
    now = datetime.now(pytz.timezone('US/Eastern'))
    current_time_str = now.strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")
    start_trading_time = datetime.now(pytz.timezone('US/Eastern')).replace(hour=10, minute=2, second=0, microsecond=0)
    if datetime.now(pytz.timezone('US/Eastern')) < start_trading_time:
        print("")
        print("Exiting buy_stocks: Market trading time not yet reached.")
        return
    target_time = datetime.now(pytz.timezone('US/Eastern')).replace(hour=15, minute=56, second=0, microsecond=0)
    if datetime.now(pytz.timezone('US/Eastern')) > target_time:
        print("")
        print("Exiting buy_stocks: Outside of buy strategy times.")
        print("")
        return
    print("")
    print(f"Starting buy_stocks: Monitoring {len(stocks_to_buy)} symbols simultaneously for 3 minutes each.")
    print("")
    start_time = time.time()
    original_start_time = start_time
    with buy_sell_lock:
        price_changes = {symbol: {'increased': 0, 'decreased': 0} for symbol in stocks_to_buy}
    try:
        # Create and start a thread for each symbol
        threads = []
        for symbol in stocks_to_buy:
            thread = threading.Thread(target=track_price_changes, args=(symbol, 180))
            threads.append(thread)
            print(f"Starting monitoring thread for {symbol}...")
            thread.start()
            time.sleep(10)  # Stagger threads by 10 seconds
        # Wait for all threads to complete (3 minutes monitoring per thread)
        for thread in threads:
            thread.join()
        print("")
        print(f"Completed simultaneous monitoring of all {len(stocks_to_buy)} symbols for 3 minutes each.")
        print("")
        # Process buy decisions for all symbols
        for symbol in stocks_to_buy:
            cash_available = calculate_cash_on_hand()
            total_symbols = calculate_total_symbols(stocks_to_buy)
            allocation_per_symbol = allocate_cash_equally(cash_available, total_symbols)
            current_price = get_current_price(symbol)
            qty = round(allocation_per_symbol / current_price, 4) if current_price else 0
            total_cost_for_qty = allocation_per_symbol
            print("")
            status_printer_buy_stocks()
            print("")
            print(f"Symbol: {symbol}")
            print(f"Current Price: {current_price}")
            print(f"Estimated Quantity: {qty}")
            print(f"Total Cost for Qty: {total_cost_for_qty}")
            print("")
            print(f"Cash Available: {cash_available}")
            print("")
            print(f"Increased: {price_changes[symbol]['increased']}")
            print(f"Decreased: {price_changes[symbol]['decreased']}")
            print("")
            total_increases = price_changes[symbol]['increased']
            total_decreases = price_changes[symbol]['decreased']
            print("")
            print(f"Total Price Increases for {symbol}: {total_increases}")
            print(f"Total Price Decreases for {symbol}: {total_decreases}")
            print("")
            overall_total_increases = sum(price_changes[symbol]['increased'] for symbol in stocks_to_buy)
            overall_total_decreases = sum(price_changes[symbol]['decreased'] for symbol in stocks_to_buy)
            print("")
            print(f"Overall Total Price Increases: {overall_total_increases}")
            print(f"Overall Total Price Decreases: {overall_total_decreases}")
            print("")
            historical_data = calculate_technical_indicators(symbol, lookback_days=90)
            macd_value = historical_data['macd'].iloc[-1]
            rsi_value = historical_data['rsi'].iloc[-1]
            volume_value = historical_data['volume'].iloc[-1]
            print(f"MACD: {macd_value}")
            print(f"RSI: {rsi_value}")
            print(f"Volume: {volume_value}")
            print("")
            favorable_macd_condition = historical_data['signal'].iloc[-1] > 0.15
            favorable_rsi_condition = historical_data['rsi'].iloc[-1] > 70
            favorable_volume_condition = historical_data['volume'].iloc[-1] > 0.85 * historical_data['volume'].mean()
            if (cash_available >= total_cost_for_qty and
                    ((price_changes[symbol]['increased'] >= 3 and
                      price_changes[symbol]['increased'] > price_changes[symbol]['decreased'] and
                      favorable_macd_condition and
                      favorable_rsi_condition and
                      favorable_volume_condition) or
                     rsi_value >= 65)):
                if qty > 0:
                    print(f" ******** Buying stocks for {symbol}... ")
                    print_technical_indicators(symbol, calculate_technical_indicators(symbol))
                    api_symbol = symbol.replace('-', '.')
                    try:
                        buy_order = api.submit_order(
                            symbol=api_symbol,
                            notional=total_cost_for_qty,  # Use notional for fractional shares
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        print(f" {current_time_str}, Bought {qty:.4f} shares of {api_symbol} at {current_price:.2f} (notional: ${total_cost_for_qty:.2f})")
                        logging.info(f"{current_time_str} Buy {qty:.4f} shares of {api_symbol}.")
                        with open(csv_filename, mode='a', newline='') as csv_file:
                            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                            csv_writer.writerow({
                                'Date': current_time_str,
                                'Buy': 'Buy',
                                'Quantity': qty,
                                'Symbol': api_symbol,
                                'Price Per Share': current_price
                            })
                        stocks_to_remove.append((api_symbol, current_price, today_date_str))
                        buy_stock_green_light = 1
                        order_filled = False
                        for _ in range(30):
                            order_status = api.get_order(buy_order.id)
                            if order_status.status == 'filled':
                                order_filled = True
                                break
                            time.sleep(2)
                        if order_filled and api.get_account().daytrade_count < 3:
                            stop_order_id = place_trailing_stop_sell_order(api_symbol, qty, current_price)
                            if stop_order_id:
                                print(f"Trailing stop sell order placed for {api_symbol} with ID: {stop_order_id}")
                            else:
                                print(f"Failed to place trailing stop sell order for {api_symbol}")
                        else:
                            print(f"Buy order not filled or day trade limit reached for {api_symbol}")
                    except Exception as e:
                        print(f"Error submitting buy order for {api_symbol}: {e}")
                        logging.error(f"Error submitting buy order for {api_symbol}: {e}")
                else:
                    print("Price increases or RSI favorable, but quantity is 0. Not buying.")
                    buy_stock_green_light = 0
            else:
                print("Not buying based on technical indicators or price decreases.")
                print_technical_indicators(symbol, calculate_technical_indicators(symbol))
            time.sleep(2)
    except Exception as e:
        print(f"An error occurred in buy_stocks: {str(e)}")
        logging.error(f"An error occurred in buy_stocks: {str(e)}")
    try:
        with buy_sell_lock:
            for symbol, price, date in stocks_to_remove:
                bought_stocks[symbol] = (round(price, 4), date)
                stocks_to_buy.remove(symbol.replace('.', '-'))
                remove_symbol_from_trade_list(symbol.replace('.', '-'))
                trade_history = TradeHistory(
                    symbol=symbol,
                    action='buy',
                    quantity=qty,
                    price=price,
                    date=date
                )
                session.add(trade_history)
                db_position = Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_price=price,
                    purchase_date=date
                )
                session.add(db_position)
            session.commit()
            refresh_after_buy()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Database error: {str(e)}")
        logging.error(f"Database error: {str(e)}")
    print("")
    print("Completed buy_stocks processing for all symbols.")
    print("")

def refresh_after_buy():
    global stocks_to_buy, bought_stocks
    time.sleep(2)
    stocks_to_buy = get_stocks_to_trade()
    bought_stocks = update_bought_stocks_from_api()

def place_trailing_stop_sell_order(symbol, qty, current_price):
    try:
        # Check if qty is a fractional share
        if qty != int(qty):
            print(f"Skipping trailing stop sell order for {symbol}: Fractional share quantity {qty:.4f} detected.")
            logging.error(f"Skipped trailing stop sell order for {symbol}: Fractional quantity {qty:.4f} not allowed.")
            return None

        stop_loss_percent = 1.0
        stop_loss_price = current_price * (1 - stop_loss_percent / 100)
        print(f"Attempting to place trailing stop sell order for {qty} shares of {symbol} "
              f"with trail percent {stop_loss_percent}% (initial stop price: {stop_loss_price:.2f})")

        stop_order = api.submit_order(
            symbol=symbol,
            qty=int(qty),
            side='sell',
            type='trailing_stop',
            trail_percent=str(stop_loss_percent),
            time_in_force='gtc'
        )

        print(f"Successfully placed trailing stop sell order for {qty} shares of {symbol} "
              f"with ID: {stop_order.id}")
        logging.info(f"Placed trailing stop sell order for {qty} shares of {symbol} with ID: {stop_order.id}")

        return stop_order.id

    except Exception as e:
        print(f"Error placing trailing stop sell order for {symbol}: {str(e)}")
        logging.error(f"Error placing trailing stop sell order for {symbol}: {str(e)}")
        return None

def update_bought_stocks_from_api():
    positions = api.list_positions()
    bought_stocks = {}

    for position in positions:
        symbol = position.symbol
        avg_entry_price = float(position.avg_entry_price)
        quantity = float(position.qty)  # Use float to handle fractional shares

        # Fetch the most recent filled buy order date using list_orders
        try:
            # Set initial end_time to 30 days prior to ensure historical orders
            end_time = (datetime.now(pytz.UTC) - timedelta(days=30)).isoformat()
            order_list = []
            CHUNK_SIZE = 500

            while True:
                order_chunk = api.list_orders(
                    status='all',
                    nested=False,
                    direction='desc',
                    until=end_time,
                    limit=CHUNK_SIZE
                )
                if order_chunk:
                    order_list.extend(order_chunk)
                    end_time = (order_chunk[-1].submitted_at - timedelta(seconds=1)).isoformat()
                else:
                    break

            # Find the most recent filled buy order for the symbol
            buy_order = None
            for order in order_list:
                if order.symbol == symbol and order.side == 'buy' and order.status == 'filled' and order.submitted_at:
                    buy_order = order
                    break

            if buy_order:
                purchase_date = buy_order.submitted_at.date()
                purchase_date_str = purchase_date.strftime("%Y-%m-%d")
                print(f"Fetched purchase date for {symbol}: {purchase_date_str}")
                logging.info(f"Fetched purchase date for {symbol}: {purchase_date_str}")
            else:
                # If position exists but no buy order found, use yesterday's date to enable selling
                purchase_date = datetime.today().date() - timedelta(days=1)
                purchase_date_str = purchase_date.strftime("%Y-%m-%d")
                print(f"No filled buy order found for {symbol}. Using yesterday's date: {purchase_date_str} to enable selling")
                logging.info(f"No filled buy order found for {symbol}. Using yesterday's date: {purchase_date_str} to enable selling")
                logging.warning(f"No buy orders found for {symbol}. Orders fetched: {len(order_list)}. Check Alpaca API data or order history.")

        except Exception as e:
            purchase_date = datetime.today().date() - timedelta(days=1)
            purchase_date_str = purchase_date.strftime("%Y-%m-%d")
            print(f"Error fetching buy orders for {symbol}: {e}. Using yesterday's date: {purchase_date_str} to enable selling")
            logging.error(f"Error fetching buy orders for {symbol}: {e}. Using yesterday's date: {purchase_date_str} to enable selling")

        try:
            db_position = session.query(Position).filter_by(symbol=symbol).one()
            db_position.quantity = quantity
            db_position.avg_price = avg_entry_price
            db_position.purchase_date = purchase_date_str
        except NoResultFound:
            db_position = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=avg_entry_price,
                purchase_date=purchase_date_str
            )
            session.add(db_position)

        bought_stocks[symbol] = (avg_entry_price, purchase_date_str)

    session.commit()
    return bought_stocks

def sell_stocks(bought_stocks, buy_sell_lock):
    stocks_to_remove = []
    now = datetime.now(pytz.timezone('US/Eastern'))
    current_time_str = now.strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")
    today_date_str = datetime.today().date().strftime("%Y-%m-%d")
    comparison_date = datetime.today().date()  # Always set to today's date

    for symbol, (bought_price, purchase_date) in bought_stocks.items():
        status_printer_sell_stocks()
        
        # Parse purchase_date as a date object
        try:
            bought_date = datetime.strptime(purchase_date, "%Y-%m-%d").date()
        except (ValueError, TypeError) as e:
            print(f"Error parsing purchase_date for {symbol}: {purchase_date}. Skipping. Error: {e}")
            logging.error(f"Error parsing purchase_date for {symbol}: {purchase_date}. Error: {e}")
            continue

        # Log date comparison details
        print(f"Checking {symbol}: Purchase date = {bought_date}, Comparison date = {comparison_date}")
        logging.info(f"Checking {symbol}: Purchase date = {bought_date}, Comparison date = {comparison_date}")

        if bought_date < comparison_date:  # Strict < to prevent same-day trading
            current_price = get_current_price(symbol)
            if current_price is None:
                print(f"Skipping {symbol}: Could not retrieve current price.")
                logging.error(f"Skipping {symbol}: Could not retrieve current price.")
                continue

            try:
                position = api.get_position(symbol)
                bought_price = float(position.avg_entry_price)
                qty = float(position.qty)  # Use float for fractional shares

                # Fetch all open orders and filter for the specific symbol
                open_orders = api.list_orders(status='open', symbols=symbol)
                symbol_open_orders = [order for order in open_orders if order.symbol == symbol]
                
                print(f"{symbol}: Found {len(symbol_open_orders)} open orders.")
                logging.info(f"{symbol}: Found {len(symbol_open_orders)} open orders.")
                
                if symbol_open_orders:
                    print(f"There is an open sell order for {symbol}. Skipping sell order.")
                    logging.info(f"{current_time_str} Skipped sell for {symbol} due to existing open order.")
                    continue

                # Check sell condition: current price is at least 0.5% higher than bought price
                sell_threshold = bought_price * 1.005
                print(f"{symbol}: Current price = {current_price:.2f}, Bought price = {bought_price:.2f}, Sell threshold = {sell_threshold:.2f}")
                logging.info(f"{symbol}: Current price = {current_price:.2f}, Bought price = {bought_price:.2f}, Sell threshold = {sell_threshold:.2f}")

                if current_price >= sell_threshold:
                    api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    print(f" {current_time_str}, Sold {qty:.4f} shares of {symbol} at {current_price:.2f} based on a higher selling price.")
                    logging.info(f"{current_time_str} Sold {qty:.4f} shares of {symbol} at {current_price:.2f}.")
                    
                    with open(csv_filename, mode='a', newline='') as csv_file:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                        csv_writer.writerow({
                            'Date': current_time_str,
                            'Sell': 'Sell',
                            'Quantity': qty,
                            'Symbol': symbol,
                            'Price Per Share': current_price
                        })
                    stocks_to_remove.append((symbol, qty, current_price))
                    time.sleep(2)
                else:
                    print(f"{symbol}: Price condition not met. Current price ({current_price:.2f}) < Sell threshold ({sell_threshold:.2f})")
                    logging.info(f"{symbol}: Price condition not met. Current price ({current_price:.2f}) < Sell threshold ({sell_threshold:.2f})")
                time.sleep(2)
            except Exception as e:
                print(f"Error processing sell for {symbol}: {e}")
                logging.error(f"Error processing sell for {symbol}: {e}")
        else:
            print(f"{symbol}: Not eligible for sale. Purchase date ({bought_date}) is not before comparison date ({comparison_date})")
            logging.info(f"{symbol}: Not eligible for sale. Purchase date ({bought_date}) is not before comparison date ({comparison_date})")

    try:
        with buy_sell_lock:
            for symbol, qty, current_price in stocks_to_remove:
                del bought_stocks[symbol]
                trade_history = TradeHistory(
                    symbol=symbol,
                    action='sell',
                    quantity=qty,
                    price=current_price,
                    date=today_date_str
                )
                session.add(trade_history)
                session.query(Position).filter_by(symbol=symbol).delete()
            session.commit()
            refresh_after_sell()
    except SQLAlchemyError as e:
        session.rollback()
        print(f"Database error: {str(e)}")
        logging.error(f"Database error: {str(e)}")

def refresh_after_sell():
    global bought_stocks
    time.sleep(2)
    bought_stocks = update_bought_stocks_from_api()

def load_positions_from_database():
    positions = session.query(Position).all()
    bought_stocks = {}
    for position in positions:
        symbol = position.symbol
        avg_price = position.avg_price
        purchase_date = position.purchase_date
        bought_stocks[symbol] = (avg_price, purchase_date)
    return bought_stocks

def main():
    global stocks_to_buy
    stocks_to_buy = get_stocks_to_trade()
    bought_stocks = load_positions_from_database()
    buy_sell_lock = threading.Lock()
    while True:
        try:
            stop_if_stock_market_is_closed()
            now = datetime.now(pytz.timezone('US/Eastern'))
            current_time_str = now.strftime("Eastern Time | %I:%M:%S %p | %m-%d-%Y |")
            cash_balance = round(float(api.get_account().cash), 2)
            print("------------------------------------------------------------------------------------")
            print("2025 Edition of the Bull Market Advanced Stock Market Trading Robot, Version 8 ")
            print("by https://github.com/CodeProSpecialist")
            print("------------------------------------------------------------------------------------")
            print(f"  {current_time_str} Cash Balance: ${cash_balance}")
            day_trade_count = api.get_account().daytrade_count
            print("\n")
            print(f"Current day trade number: {day_trade_count} out of 3 in 5 business days")
            print("\n")
            print("------------------------------------------------------------------------------------")
            print("\n")
            stocks_to_buy = get_stocks_to_trade()
            if not bought_stocks:
                bought_stocks = update_bought_stocks_from_api()
            sell_thread = threading.Thread(target=sell_stocks, args=(bought_stocks, buy_sell_lock))
            buy_thread = threading.Thread(target=buy_stocks, args=(bought_stocks, stocks_to_buy, buy_sell_lock))
            sell_thread.start()
            buy_thread.start()
            sell_thread.join()
            buy_thread.join()
            if PRINT_STOCKS_TO_BUY:
                print("\n")
                print("------------------------------------------------------------------------------------")
                print("\n")
                print("Stocks to Purchase:")
                print("\n")
                for symbol in stocks_to_buy:
                    current_price = get_current_price(symbol)
                    print(f"Symbol: {symbol} | Current Price: {current_price} ")
                    time.sleep(1)
                print("\n")
                print("------------------------------------------------------------------------------------")
                print("\n")
            if PRINT_ROBOT_STORED_BUY_AND_SELL_LIST_DATABASE:
                print_database_tables()
            if DEBUG:
                print("\n")
                print("------------------------------------------------------------------------------------")
                print("\n")
                print("Stocks to Purchase:")
                print("\n")
                for symbol in stocks_to_buy:
                    current_price = get_current_price(symbol)
                    atr_low_price = get_atr_low_price(symbol)
                    print(f"Symbol: {symbol} | Current Price: {current_price} | ATR low buy signal price: {atr_low_price}")
                print("\n")
                print("------------------------------------------------------------------------------------")
                print("\n")
                print("\nStocks to Sell:")
                print("\n")
                for symbol, _ in bought_stocks.items():
                    current_price = get_current_price(symbol)
                    atr_high_price = get_atr_high_price(symbol)
                    print(f"Symbol: {symbol} | Current Price: {current_price} | ATR high sell signal profit price: {atr_high_price}")
                print("\n")
            print("Do Not Stop this Robot or you will need to ")
            print("delete the trading_bot.db database file and start over again with an empty database. ")
            print("Placing trades without this Robot will also require ")
            print("deleting the trading_bot.db database file and starting over again with an empty database. ")
            print("")
            print("Waiting 60 seconds before checking price data again........")
            print("")
            time.sleep(60)
        except Exception as e:
            logging.error(f"Error encountered: {e}")
            time.sleep(120)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"Error encountered: {e}")
        session.close()
