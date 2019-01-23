import pandas as pd
import numpy as np

n = 10000

x1 = np.array([i / n for i in range(n)])
x2 = np.array([i / n for i in range(n)])
y = x1 + x2 + x1 ** 2
df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
df.to_csv('data/toy.csv')