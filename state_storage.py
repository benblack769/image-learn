import tensorflow as tf
import numpy as np
import random as rand
from collections import defaultdict

class StateData:
    def __init__(self,input,randvec,action,true_reward):
        self.input = input
        self.randvec = randvec
        self.action = action
        self.true_reward = true_reward


class QueueSampler:
    def __init__(self,max_size):
        assert max_size > 0
        self.max_size = max_size
        self.data = [None]*max_size*2
        self.start_idx = 0
        self.end_idx = 0

    def size(self):
        return self.end_idx - self.start_idx

    __len__ = size

    def full(self):
        return self.size() == self.max_size

    def add(self,item):
        data = self.data
        start_idx = self.start_idx
        end_idx = self.end_idx

        if end_idx >= len(data):
            data = data[start_idx:] + [None]*start_idx
            end_idx = len(data) - start_idx
            start_idx = 0

        data[end_idx] = item
        end_idx += 1
        if end_idx - start_idx > self.max_size:
            data[start_idx] = None
            start_idx += 1

        self.data = data
        self.start_idx = start_idx
        self.end_idx = end_idx

    def sample_idxs(self,diff_start,diff_end,number_to_sample):
        sample_start = self.start_idx + diff_start
        sample_end = self.end_idx - diff_end
        if sample_end - sample_start <= 0:
            return None
        else:
            return np.random.randint(sample_start,sample_end,size=number_to_sample)

    def last_idx(self):
        return self.end_idx - 1

    def get_idxs(self,idxs):
        arr = self.data
        return [arr[idx] for idx in idxs]

    def get_idx(self,idx):
        return self.data[idx]

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
