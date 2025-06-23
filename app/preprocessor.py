import numpy as np
import pandas as pd
import json

with open("app/minmax_dict.json",'r') as f:
    minmax_dict=json.load(f)

indices_to_drop=[ 0,  1,  4,  5,  9, 10, 14, 20, 22, 23]
actuals_columns_list=['os1', 'os2', 's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13',
       's14', 's15', 's17', 's20', 's21']
def remove_values(df):
    """ Remove the values of sensors that were droped during training"""
    arr = np.delete(np.array(df), indices_to_drop, axis=1)
    return pd.DataFrame(arr, columns=actuals_columns_list)
def smooth(s, b = 0.98):

    v = np.zeros(len(s)+1) #v_0 is already 0.
    bc = np.zeros(len(s)+1)

    for i in range(1, len(v)): #v_t = 0.95
        v[i] = (b * v[i-1] + (1-b) * s[i-1]) 
        bc[i] = 1 - b**i

    sm = v[1:] / bc[1:]
    
    return sm
def process_data(df):
    data=remove_values(df)

    # Minmax Scaling of input data
    for c in data.columns:
        if 's' in c:
            data[c] = (data[c] - minmax_dict[c+'min']) / (minmax_dict[c+'max'] - minmax_dict[c+'min'])

    # Smothing
    for c in data.columns:
    
        if 's' in c:
            sm_list = []

            s = np.array(data[c].copy())
            sm = list(smooth(s, 0.98))
            sm_list += sm
            
            data[c+'_smoothed'] = sm_list
    #Remove the original series
    for c in data.columns:
        if ('s' in c) and ('smoothed' not in c):
            data[c] = data[c+'_smoothed']
            data.drop(c+'_smoothed', axis = 1, inplace = True)
    return data