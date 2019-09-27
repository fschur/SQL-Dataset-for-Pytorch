import time
from torch.utils.data.dataloader import DataLoader
import sqlite3
import numpy as np
from dataset_sql import SQLDatasetPreload, SQLDataset


"""
Creating an SQLlite databse in order to compare performance between the two datasets, when iterating over the database.
"""

conn = sqlite3.connect('test.db')
c = conn.cursor()

c.execute("CREATE TABLE example_table (id INTEGER PRIMARY KEY AUTOINCREMENT, x1 REAL, x2 REAL, y REAL)")
conn.commit()

data = np.random.random((100000, 3))
i = 0
for obs in data:
    if i % 1000 == 0:    
        print(i)
    i += 1
    c.execute("INSERT INTO example_table (x1, x2, y) VALUES (?,?,?)", tuple(obs))
    conn.commit()

conn.close()


# Performance when data is not shuffled
# and preloaded
st = time.time()
dataset = SQLDatasetPreload("test.db", "example_table", 10000, False)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128)

for (x, label) in dataloader:
    continue

end = time.time()
performance_preload = end-st
print("Peformance when data is not shuffled and preloaded (in s): ", performance_preload)

# and not preloaded
st = time.time()
dataset = SQLDataset("test.db", "example_table")
dataloader = DataLoader(dataset, shuffle=False, batch_size=128)

for (x, label) in dataloader:
    continue

end = time.time()
performance = end-st
print("Peformance when data is not shuffled and not preloaded (in s): ", performance)

print("Preloading is ", performance/performance_preload, " times faster. \n")


# Performance when data is shuffled
# and preloaded
st = time.time()
dataset = SQLDatasetPreload("test.db", "example_table", 10000, True)
dataloader = DataLoader(dataset, shuffle=False, batch_size=128)

for (x, label) in dataloader:
    continue

end = time.time()
performance_preload = end-st
print("Peformance when data is shuffled and preloaded (in s): ", performance_preload)

# and not preloaded
st = time.time()
dataset = SQLDataset("test.db", "example_table")
dataloader = DataLoader(dataset, shuffle=True, batch_size=128)

for (x, label) in dataloader:
    continue

end = time.time()
performance = end-st
print("Peformance when data is shuffled and not preloaded (in s): ", performance)

print("Preloading is ", performance/performance_preload, " times faster.")
