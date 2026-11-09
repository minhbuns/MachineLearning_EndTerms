import pandas as pd
import numpy as np

data = pd.read_csv('online_shoppers_intention.csv')

print('Dataset information:')
data_len = data.shape[0]
print(f'Quantity: {data_len}')
cols = data.columns
print(f'Number of columns: {len(cols)}')
print('Short description:')
print('Col --- datatype --- unique_value')
print('-' * 30)

numeric_key_word = ['Duration', 'Rates', 'Values']

for c in cols:
    name = c
    datatype = 'num' if data[c].dtype in ('int64', 'float64') else data[c].dtype
    number_of_value = len(data[c].unique())
    stype = 'numeric' if any(map(lambda x: x in c, numeric_key_word)) else 'category'
    print(f'{name} --- {datatype} --- {number_of_value} --- {stype}')
    
