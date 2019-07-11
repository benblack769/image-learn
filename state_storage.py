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

class Sampler:
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


def gather_item_idxs(items,idxs):
    return [item[idx] for item,idx in zip(items,idxs)]

class TempStorage:
    def __init__(self,keep_size):
        self.keep_size = keep_size
        self.data = []

    def add(self,value,item):
        if len(self.data) < self.keep_size:
            self.data.append([value,item])
        else:
            self._add_update(value,item)

    def _add_update(self,value,item):
        min_value = value
        NEW_MIN_IDX = -1
        min_idx = NEW_MIN_IDX
        CONSIZER_SIZE = 3
        for s_idx in self.sample_idxs(CONSIZER_SIZE):
            samp_data = self.data[s_idx]
            val = samp_data[0]
            if val >= min_value:
                min_value = val
                min_idx = s_idx

        if min_idx != NEW_MIN_IDX:
            self.data[min_idx] = [value,item]


    def update(self,value,idx):
        self.data[idx][0] = value

    def sample_idxs(self,number_idxs):
        return np.random.choice(len(self.data),replace=False,size=number_idxs)

    def get_idxs(self,idxs):
        return [self.data[i][1] for i in idxs]

    def full(self):
        return self.keep_size <= len(self.data)

def combine_dict_list(dict_list):
    list_dict = defaultdict(list)
    for dict in dict_list:
        for k,v in dict.items():
            list_dict[k].append(v)
    return list_dict

class StateStorage:
    def __init__(self,num_envs,keep_size):
        self.keep_size = keep_size
        self.num_envs = num_envs
        QUEUE_SIZE = 3
        self.queue = QueueSampler(QUEUE_SIZE)
        self.sampler = TempStorage(keep_size)

    '''def sample_train_data(self,batch_size):
        assert len(self.queue) == self.keep_size

        sampled_idxs = self.queue.sample_idxs(1,1,batch_size)
        env_idxs = np.random.randint(0,self.num_envs,size=batch_size)

        prev_entry = gather_item_idxs(self.queue.get_idxs(sampled_idxs - 1),env_idxs)
        input_entry = gather_item_idxs(self.queue.get_idxs(sampled_idxs),env_idxs)
        next_entry = gather_item_idxs(self.queue.get_idxs(sampled_idxs + 1),env_idxs)
        return {
            "prev_input": np.stack([pe.input for pe in prev_entry]),
            "input": np.stack([pe.input for pe in input_entry]),
            "next_input": np.stack([pe.input for pe in next_entry]),
            "action": np.stack([pe.action for pe in input_entry]),
            "cur_randvec": np.stack([pe.randvec for pe in input_entry]),
            "prev_randvec": np.stack([pe.randvec for pe in prev_entry]),
            "reward": np.stack([pe.true_reward for pe in input_entry]),
        }'''

    def update_data(self,idxs):
        num_entries = len(entries)

    def add_data(self,state_datas,advantage_magnitudes):
        self.queue.add(state_datas)

        if self.queue.full():
            last_idx = self.queue.last_idx()

            completed_idx = last_idx - 1

            prev_entry = self.queue.get_idx(completed_idx - 1)
            input_entry = self.queue.get_idx(completed_idx)
            next_entry = self.queue.get_idx(completed_idx + 1)
            new_entries = [{
                "prev_input": pe.input,
                "input": ie.input,
                "next_input": ne.input,
                "action": ie.action,
                "cur_randvec": ie.randvec,
                "prev_randvec": pe.randvec,
                "next_randvec": ne.randvec,
                "next_action": ne.action,
                "reward": ie.true_reward,
            } for pe,ie,ne in zip(prev_entry,input_entry,next_entry)]
            for mag,entry in zip(advantage_magnitudes,new_entries):
                self.sampler.add(mag,entry)

    def sample_idxs(self,size):
        return self.sampler.sample_idxs(size)

    def get_idxs(self,idxs):
        return combine_dict_list(self.sampler.get_idxs(idxs))

    def update_idxs(self,idxs,advantage_magnitudes):
        for idx,mag in zip(idxs,advantage_magnitudes):
            return self.sampler.update(mag,idx)

    def should_sample(self):
        return self.sampler.full()
