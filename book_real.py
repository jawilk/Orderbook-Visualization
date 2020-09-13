import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import numpy as np

from copy import deepcopy
import math
import pytz
import os

import book_opt


starts = [0,4,8,12,16,20,10] 
ends = [4,8,12,16,20,23,20]
snapshots = []
DATE = '09-13'
PAIR = 'FTX-MTA-PERP'
PATH = f'data/book/'

for s, e, snapshot in zip(starts, ends, snapshots):

    END = pd.Timestamp(f'2020-09-13 {e}:00:00', tz=None)

    DELTAS_PATH = f'data/deltas/book-FTX-MTA-PERP/FTX_MTA-PERP_delta_2020-{DATE} {s}:00:00_2020-{DATE} {e}:00:00.csv'
    deltas = pd.read_csv(DELTAS_PATH, index_col=0)
    deltas.index = pd.to_datetime(deltas.index)
    TOTAL_UPDATES = len(np.unique(deltas.index))

    Book = book_opt.BookState(f'data/snapshots/book-FTX-MTA-PERP/{snapshot}')
    timestamp = Book.timestamps[0]

    bid_book, ask_book = {}, {}   
    bid_book = book_opt.extract_levels(Book.bids, bid_book, count=0)
    ask_book = book_opt.extract_levels(Book.asks, ask_book, count=0)

    best_bids = np.zeros(TOTAL_UPDATES)
    best_asks = np.zeros(TOTAL_UPDATES)
    mid_price = np.zeros(TOTAL_UPDATES)
    best_bids[0] = Book.best_bid
    best_asks[0] = Book.best_ask
    mid_price[0] = (Book.best_ask + Book.best_bid) / 2

    count = book_opt.ingest_delta(Book, deltas, bid_book, ask_book, best_bids, best_asks, mid_price, END, timestamp)

    ys = []
    xmins = []
    xmaxs = []
    colors = []

    ys, xmins, xmaxs, colors = book_opt.get_plot_values(bid_book, ys, xmins, xmaxs, colors, 4.3, True, count)
    ys, xmins, xmaxs, colors = book_opt.get_plot_values(ask_book, ys, xmins, xmaxs, colors, 4.8, False, count)

    plt.figure(figsize=(30,25))
    plt.hlines(ys, xmins, xmaxs, colors)
    plt.plot(mid_price[:count], c='yellow', linewidth=1)
    plt.plot(best_bids[:count], c='green')
    plt.plot(best_asks[:count], c='red')
    plt.savefig(f'{PATH}{PAIR}_{DATE}_{s}_{e}.png')
