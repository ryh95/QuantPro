import datetime
import numpy as np

from backtest import Backtest
from data import HistoricCSVDataHandler
from event import SignalEvent
from execution import SimulatedExecutionHandler
from portfolio import Portfolio
from strategy import Strategy, MovingAverageCrossStrategy

if __name__ == "__main__":
    # csv_dir = REPLACE_WITH_YOUR_CSV_DIR_HERE
    csv_dir = './'
    symbol_list = ['AAPL']
    initial_capital = 100000.0
    start_date = datetime.datetime(1990,1,1,0,0,0)
    heartbeat = 0.0

    backtest = Backtest(csv_dir, 
                        symbol_list, 
                        initial_capital, 
                        heartbeat,
                        start_date,
                        HistoricCSVDataHandler, 
                        SimulatedExecutionHandler, 
                        Portfolio, 
                        MovingAverageCrossStrategy)
    
    backtest.simulate_trading()
