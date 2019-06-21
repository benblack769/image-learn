import tensorflow as tf
import numpy as np
import random as rand

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

    def get_idxs(self,idxs):
        arr = self.data
        return [arr[idx] for idx in idxs]

def gather_item_idxs(items,idxs):
    return [item[idx] for item,idx in zip(items,idxs)]

class StateStorage:
    def __init__(self,num_envs,keep_size):
        self.keep_size = keep_size
        self.num_envs = num_envs
        self.queue = QueueSampler(self.keep_size)

    def sample_train_data(self,batch_size):
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
        }

    '''def sample_randvec_data(self):
        assert len(self.datas) == self.keep_size

        learn_idx = self.keep_size-1#rand.randrange(1,)
        prev_input_entry = self.datas[learn_idx-1]
        input_entry = self.datas[learn_idx]
        reward_entry = self.datas[learn_idx+1]
        return {
            "prev_input": prev_input_entry.input,
            "input": input_entry.input,
        }'''

    '''def sample_actor_data(self):
        assert len(self.datas) == self.keep_size

        learn_idx = rand.randrange(0,self.keep_size)
        input_entry = self.datas[learn_idx]
        return {
            "randvec": input_entry.randvec,
            "input": input_entry.input,
        }'''

    def add_data(self,state_datas):
        self.queue.add(state_datas)

    def should_sample(self):
        return len(self.queue) == self.keep_size
