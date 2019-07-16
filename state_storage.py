import tensorflow as tf
import numpy as np
import random as rand
from collections import defaultdict

class StateData:
    def __init__(self,input,action,true_reward):
        self.input = input
        self.action = action
        self.true_reward = true_reward

    def listify(self):
        return [self.input,self.action,self.true_reward]

    @classmethod
    def from_list(cls,l):
        return StateData(*l)

class StoredData:
    def __init__(self,prev_input,cur_input,next_input,action,true_reward):
        self.prev_input = prev_input
        self.cur_input = cur_input
        self.next_input = next_input
        self.action = action
        self.true_reward = true_reward

    def listify(self):
        return [
            self.prev_input,
            self.cur_input,
            self.next_input,
            self.action,
            self.true_reward,
        ]

    @classmethod
    def from_list(cls,l):
        return StoredData(*l)

def concat_stores(data1,data2):
    return StoredData.from_list([np.concatenate([d1,d2],axis=0) for d1,d2 in zip(data1.listify(),data2.listify())])

def sample_data_by_weights(store_data,weights,sample_size):
    weights = np.reshape(weights,[len(weights)])
    sum = np.sum(weights)
    idxs = np.random.choice(len(weights),size=sample_size,p=weights/sum,replace=False)
    return StoredData.from_list([d[idxs] for d in store_data.listify()])

class StateQueue:
    KEEP_SIZE = 3

    def __init__(self):
        self.data = []

    def add(self,state_data):
        if len(self.data) >= self.KEEP_SIZE:
            self.data.pop(0)

        self.data.append(state_data)

    def should_get(self):
        return len(self.data) >= self.KEEP_SIZE

    def get(self):
        return StoredData(
            prev_input=self.data[0].input,
            cur_input=self.data[1].input,
            next_input=self.data[2].input,
            action=self.data[1].action,
            true_reward=self.data[1].true_reward,
        )

class IterStorage:
    def __init__(self):
        self.state_queue = StateQueue()
        self.stored_datas = StoredData([],[],[],[],[])

    def add(self,new_datas):
        self.state_queue.add(new_datas)
        if self.state_queue.should_get():
            new_storage_data = self.state_queue.get()
            for data_list, new_d in zip(self.stored_datas.listify(),new_storage_data.listify()):
                data_list.append(new_d)

    def consoladate(self):
        data = StoredData.from_list([np.concatenate(datas,axis=0) for datas in self.stored_datas.listify()])
        return data

class DataSampler:
    def __init__(self,storage_data):
        self.sample_idx = 0
        self.data = storage_data
        self.shuffle()

    def get_batch(self,batch_size):
        if self.sample_idx + batch_size > self.size():
            self.sample_idx = 0
            self.shuffle()

        start = self.sample_idx
        batch = StoredData.from_list([d[start:start + batch_size] for d in self.data.listify()])
        self.sample_idx += batch_size
        return batch

    def size(self):
        return len(self.data.cur_input)

    def shuffle(self):
        idx_array = np.arange(self.size())
        np.random.shuffle(idx_array)
        self.data = StoredData.from_list([d[idx_array] for d in self.data.listify()])
