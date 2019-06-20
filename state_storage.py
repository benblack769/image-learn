import tensorflow as tf
import numpy as np
import random as rand

class StateData:
    def __init__(self,input,randvec,action,true_reward):
        self.input = input
        self.randvec = randvec
        self.action = action
        self.true_reward = true_reward

class StateStorage:
    def __init__(self):
        self.keep_size = 3
        self.datas = []

    def sample_train_data(self):
        assert len(self.datas) == self.keep_size

        learn_idx = rand.randrange(1,self.keep_size-1)
        prev_input_entry = self.datas[learn_idx-1]
        input_entry = self.datas[learn_idx]
        reward_entry = self.datas[learn_idx+1]
        return {
            "prev_input": prev_input_entry.input,
            "input": input_entry.input,
            "next_input": reward_entry.input,
            "action": input_entry.action,
            "cur_randvec": input_entry.randvec,
            "prev_randvec": prev_input_entry.randvec,
            "reward": input_entry.true_reward,
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

    def add_data(self,state_data):
        self.datas.append(state_data)
        if len(self.datas) > self.keep_size:
            self.datas.pop(0)

    def should_sample(self):
        return len(self.datas) == self.keep_size
