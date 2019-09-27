import torch
from torch.utils.data.dataset import Dataset
import sqlite3
import random

"""
Two Pytorch Dataset to iterate over a SQLlite database. Instead of loading the observations one by one as done by
SQLDataset, SQLDatasetPreload loads full chunks of data at once. This speeds up the iteration process a lot.
Note that when using SQLDatasetPreload, shuffling needs to be set to false when using the dataloader. If shuffling is
needed, set shuffle = True when initializing SQLDatasetPreload. It is also assumed that the first column of the 
database is called id and starts at 1 and ends at len(databse).
"""


class SQLDatasetPreload(Dataset):
    def __init__(self, db_name, table_name, buffer_size, shuffle):
        # connect to db
        self.conn = sqlite3.connect(db_name)
        self.table_name = table_name
        self.c = self.conn.cursor()
        # get the length via the id
        self.c.execute("SELECT count(*) FROM " + self.table_name)
        self.len = self.c.fetchall()[0][0]
        self.c.execute("SELECT * FROM " + self.table_name + " WHERE id = 1")
        self.obs_len = len(self.c.fetchall()[0])-1
        # initialize the buffer for preloading
        self.buffer_size = buffer_size
        self.buffer_num = 0
        self.shuffle = shuffle
        if shuffle:
            self.idx_left = [i for i in range(1, self.len+1)]
        self.new_buffer()

    def new_buffer(self):
        # loads new data
        # if shuffle == True, then get random observation that have not been seen before.
        if self.shuffle:
            idx = random.sample(self.idx_left, min(self.buffer_size, len(self.idx_left)))
            self.idx_left = list(set(self.idx_left) - set(idx))
        else:
            idx = [i+1 for i in range(self.buffer_num*self.buffer_size, (self.buffer_num+1)*self.buffer_size)]

        self.c.execute("SELECT * FROM " + self.table_name + " WHERE id IN " + str(tuple(idx)))
        self.buffer = self.c.fetchall()

    def __getitem__(self, item):
        # convert the index for the hole data to the corresponding index for the loaded data
        item = item - self.buffer_num * self.buffer_size
        # If all observations in the loaded data have been used, new data is loaded.
        if item == self.buffer_size:
            self.new_buffer()
            self.buffer_num += 1
            item = 0
        return torch.Tensor(self.buffer[item][1:self.obs_len]), torch.Tensor([self.buffer[item][self.obs_len]])

    def __len__(self):
        return self.len


class SQLDataset(Dataset):
    def __init__(self, db_name, table_name):
        # connect to db
        self.conn = sqlite3.connect(db_name)
        self.table_name = table_name
        self.c = self.conn.cursor()
        # get the length via the id
        self.c.execute("SELECT count(*) FROM " + self.table_name)
        self.len = self.c.fetchall()[0][0]
        self.c.execute("SELECT * FROM " + self.table_name + " WHERE id = 1")
        self.obs_len = len(self.c.fetchall()[0])-1

    def __getitem__(self, item):
        # get the observation directly from the db
        self.c.execute("SELECT * FROM " + self.table_name + " WHERE id = " + str(item+1))
        tmp = self.c.fetchall()
        return torch.Tensor(tmp[0][1:self.obs_len]), torch.Tensor([tmp[0][self.obs_len]])

    def __len__(self):
        return self.len
