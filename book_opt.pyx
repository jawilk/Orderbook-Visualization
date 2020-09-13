from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import numpy as np

import math
import pytz
import os

#cimport numpy as np


cdef class BookState():
    cdef:
        public dict bids
        public double best_bid
        
        public dict asks
        public double best_ask
        
        public list timestamps
        
        public list volume_imbalance
        public list spread
        public list mid_price
        
        readonly int lookback        

    SNAPSHOT_PATH = 'data/snapshots/book-FTX-MTA-PERP'
    
    def __init__(self, path_to_snapshot):
        
        self.bids = {}         
        self.best_bid = 0
        
        self.asks = {}        
        self.best_ask = 9999
        
        self.timestamps = []
        
        # Features
        self.volume_imbalance = []                                   # At best bid/ask
        self.spread = []
        self.mid_price = []
        
        self._init_book(path_to_snapshot)      
        
    def _get_snapshot(self, path_to_snapshot=None):
        if path_to_snapshot:
            with open(path_to_snapshot) as f:
                data = json.load(f)
            return data
        else:
            last_timestamp = self.timestamps[-1]
            all_snapshots_name = sorted(os.listdir(self.SNAPSHOT_PATH))
            all_snapshots = [datetime.fromtimestamp(float(name[:-5])) for name in all_snapshots_name]
            snapshots_before_start = [date for date in all_snapshots if last_timestamp < date]
            if snapshots_before_start:
                snapshot = snapshots_before_start[0]
            snapshot_index = [c for c, date in enumerate(all_snapshots) if date == snapshot][0]
            with open(f'{self.SNAPSHOT_PATH}/{all_snapshots_name[snapshot_index]}') as f:
                data = json.load(f)
            return data

    
    def _init_book(self, path_to_snapshot=None):
        cdef int quantity
        data = self._get_snapshot(path_to_snapshot)
        new_timestamp = pd.to_datetime(data['timestamp'], unit='s') + pd.to_timedelta(1, unit='h')
        self.timestamps.append(new_timestamp)
        if path_to_snapshot:
            for price, quantity in data['bid'].items():
                self.bids[float(price)] = quantity
            for price, quantity in data['ask'].items():
                self.asks[float(price)] = quantity
        else:
            bids = {}
            asks = {}
            for price, quantity in data['bid'].items():
                bids[float(price)] = quantity
            for price, quantity in data['ask'].items():
                asks[float(price)] = quantity
                
            self.bids = bids
            self.asks = asks
        self.best_bid = float(next(iter(data['bid'])))
        self.best_ask = float(next(iter(data['ask'])))
        
    def _get_spread(self):
        best_bid = next(iter(self.bids[-1]))
        best_ask = next(iter(self.asks[-1]))
        self.spread.append(best_ask - best_bid)
        self.mid_price.append((best_ask + best_bid) / 2)
        
    def _get_volume_imbalance(self):
        best_bid_vol, best_ask_vol = sum(self.bids[-1].values()[:20]), sum(self.asks[-1].values()[:20])
        volume_imbalance = (best_bid_vol - best_ask_vol) / (best_bid_vol + best_ask_vol)
        self.volume_imbalance.append(volume_imbalance)
        
    cdef void _extract_levels(self, double[:] prices, double[:] quantities, dict old_levels):
        """ Add complete level removals first to the book (i.e. quantity==0) """
        cdef:
            double price
            int quantity
            size_t i
            int total = len(prices)
        for i in range(total):
            if quantities[i] == 0:
                old_levels.pop(prices[i], None)
            else:
                old_levels[prices[i]] = quantities[i]
                
    def update_book(self, object delta):
        new_timestamp = delta.index[0].tz_localize(None)
        if not (self.timestamps[0] < new_timestamp):
            print('Timestamp too old')
            return
        self.timestamps.append(new_timestamp)

        bids_new = delta[delta['is_bid']==1]#.sort_values('quantity') # Delete first
        asks_new = delta[delta['is_bid']==0]#.sort_values('quantity')
        self._extract_levels(np.array(bids_new['price']), np.array(bids_new['quantity']), self.bids)
        self._extract_levels(np.array(asks_new['price']), np.array(asks_new['quantity']), self.asks)
        cdef double max_bid = max(self.bids)
        cdef double min_ask = min(self.asks)
        if max_bid > min_ask:
            # Negative spread, request new complete snapshot
            self._init_book()
        else:
            self.best_bid = max_bid
            self.best_ask = min_ask
        
cpdef object extract_levels(dict cur_book, dict book, int count):
    cdef:
        list prices = list(cur_book.keys())
        list quantities = list(cur_book.values())
        size_t i
        int total = len(prices)
    for i in range(total):
        if prices[i] in book:
            if quantities[i] != book[prices[i]][-1][0]: # Quantity changed
                if book[prices[i]][-1][2] == -1:
                    book[prices[i]][-1][2] = count # End sequence
                book[prices[i]].append([quantities[i], count, -1]) # Start new
        else:
            book[prices[i]] = [[quantities[i], count, -1]]
    prices = list(book.keys())
    quantities = list(book.values())
    cdef size_t j
    total = len(prices)
    for j in range(total):
        if prices[j] not in cur_book:
            if quantities[j][-1][2] == -1:
                quantities[j][-1][2] = count
    return book    


cpdef int ingest_delta(object Book, object deltas, dict bid_book, dict ask_book, double[:] best_bids, double[:] best_asks, double[:] mid_price, object end, object timestamp):
    global Book, deltas, bid_book, ask_book, best_bids, best_asks, mid_price, end, timestamp
    cdef:
        int count = 1
        size_t i
        int batch = 0
        list batches = list(Counter(deltas.index).values())
        int total = len(batches)
    for i in range(total):
        delta = deltas.iloc[batch:batch+batches[i]]
        t = delta.index[0].tz_localize(None)
        if timestamp > t:
            print('Timestamp too old')
            continue
        if t >= end:
            print('End book')
            print(count)
            return count
        
        Book.update_book(delta)
        bid_book = extract_levels(Book.bids, bid_book, count)
        ask_book = extract_levels(Book.asks, ask_book, count)
        best_bids[count] = Book.best_bid
        best_asks[count] = Book.best_ask
        mid_price[count] = (Book.best_ask + Book.best_bid) / 2
        batch += batches[i]
        count += 1
    return count
#1599728391.9711435.json
#1599742767.8191323.json
#1599724787.9882104.json

def get_plot_values(book, ys, xmins, xmaxs, colors, double thresh, is_bid, int count):
    cdef double p
    cdef list q
    cdef list _q
    for p, q in book.items():
        if is_bid and p < thresh:
            continue
        if not is_bid and p > thresh:
            continue
        for _q in q:
            ys.append(p)
            xmins.append(_q[1])
            #if q[2] == -1:
             #   xmaxs.append(count)
            #else:
            if _q[2] == -1:
                xmaxs.append(count)
            else:
                xmaxs.append(_q[2])
            if _q[0] < 300 and _q[0] > 0:
                colors.append((0,0,0))
            elif _q[0] > 300 and _q[0] < 1000:
                colors.append((0,1,1))
            elif _q[0] > 1000:
                colors.append((0,0,1))
            
    return ys, xmins, xmaxs, colors
