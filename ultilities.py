# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:16:07 2020

@author: tranl
"""

import sys, os
import pandas as pd
from datetime import datetime

### Ultility functions  
def barstr(text, symbol='#', length=100, space_size=5):
    bar_size = int((length-len(text))/2)
    bar = ''.join([symbol]*(bar_size-space_size))
    space = ''.join([' ']*space_size)
    return '{:<}{}{}{}{:>}'.format(bar, space, text, space, bar)
  
def print_(s, file):
    with open(file, "a+") as f: 
        f.write('\n' + str(s)) 
    f.close()
    print(s)

def timestr(dateTime: int, end='f'):
    if end=='m': s = pd.to_datetime(dateTime, unit='ms').strftime("%y-%m-%d %H:%M")
    elif end=='s': s = pd.to_datetime(dateTime, unit='ms').strftime("%y-%m-%d %H:%M:%S")
    elif end=='f': s = pd.to_datetime(dateTime, unit='ms').strftime("%y-%m-%d %H:%M:%S:%f")[:-3]
    return s
            
