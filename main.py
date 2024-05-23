from tools import get_vn30f
import os 

current_storage = os.path.join("d:", "data")
vn30f_storage = os.path.join(current_storage, 'vn30f')

start_time = 0
now_time = 9999999999
symbol = 'VN30F1M'
symbol2 = 'VN30'

ohlc_dict = {                                                                                                                                                                                                                 
    'basis': 'mean',                                                                                                        
    'Close': 'last',                                                                                                    
    'Volume': 'sum',}

def run():
    data = get_vn30f(start_time, now_time, symbol)
    print(data.head())

if __name__ == "__main__":
    run()