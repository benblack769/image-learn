import gym
import multiprocessing as mp
import time
import math
import random
import numpy as np
#env = gym.wrappers.Monitor(env, "recording")

def new_env():
    env = gym.make('BipedalWalkerHardcore-v2')
    return env


def process_action(env,vector):
    space = env.action_space
    if isinstance(space,gym.spaces.Box):
        return vector
    elif isinstance(space,gym.spaces.Discrete):
        return np.argmax(vector)
    else:
        raise exception()

class Envs:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.example_env = new_env()
        self.envs = [new_env() for e in range(num_envs)]

        self.observations = [env.reset() for env in self.envs]
        self.rewards = [0 for e in range(num_envs)]

        self.are_news = [False for e in range(num_envs)]

    def get_observations(self):
        return np.stack(self.observations)

    def are_new(self):
        return np.stack(self.are_news)

    def get_rewards(self):
        arr = np.stack(self.rewards)
        if len(arr.shape) < 2:
            arr = arr.reshape(arr.shape[0],1)
        return arr

    #def transform_action_vec_to_value(self,action_vec):
    #    if isinstance(new_env().action_space,gym.spaces.Discrete):
    #        pass

    def observation_space(self):
        return self.example_env.observation_space.shape

    def action_space(self):
        space = self.example_env.action_space
        if isinstance(space,gym.spaces.Box):
            return space.shape
        else:
            return space.n

    def random_actions(self):
        return [env.action_space.sample() for env in self.envs]

    def manipulate_actions(self,action_vecs):
        new_actions = []
        for i in range(self.num_envs):
            env = self.envs[i]
            action = process_action(env,action_vecs[i])

            prop_randomize_invalid = 0.3
            if not env.action_space.contains(action) and random.random() < prop_randomize_invalid:
                action = env.action_space.sample()

            prop_randomize_valid = 0.1
            if env.action_space.contains(action) and random.random() < prop_randomize_valid:
                action = env.action_space.sample()

            new_actions.append(action)

        return new_actions

    def set_actions(self, action_vecs):
        for i in range(self.num_envs):
            env = self.envs[i]
            action = process_action(env,action_vecs[i])

            if not env.action_space.contains(action):
                observation = env.reset()
                min_reward = env.reward_range[0]
                reward = max(-100,min_reward)
                done = True
            else:
                observation, reward, done, info = env.step(action)

                observation = observation if not done else env.reset()

            self.observations[i] = observation
            self.rewards[i] = reward
            self.are_news[i] = done

    #def random_actions(self):
    #    return [env.action_space.sample() for env in self.envs]

def splits(number,splits):
    res = []
    num_left = number
    for split_idx in range(splits):
        splits_left = splits - split_idx
        split_size = int(math.ceil(num_left / splits_left))

        cur_idx = number - num_left
        res.append(slice(cur_idx,cur_idx + split_size))

        num_left -= split_size

    return res

class Splitter:
    def __init__(self,number,num_splits):
        self.number = number
        self.split_list = splits(number,num_splits)

    def split(self,l):
        assert len(l) == self.number
        return [l[s] for s in self.split_list]

    def counts(self):
        return [s.stop - s.start for s in self.split_list]

class MultiProcessEnvs:
    def __init__(self, num_envs):
        self.example_env = new_env()
        self.num_envs = num_envs
        self.num_procs = mp.cpu_count()

        self.pipes = [mp.Pipe() for _ in range(self.num_procs)]
        self.main_connects = [p[1] for p in self.pipes]

        self.proc_splits = Splitter(self.num_envs,self.num_procs)

        self.procs = []

        #spawn_ctx = mp.get_context("fork") #

        for pipe,env_count in zip(self.pipes,self.proc_splits.counts()):
            proc = mp.Process(target=MultiProcessEnvs.process_start,args=(pipe[0],env_count,))
            proc.start()
            self.procs.append(proc)

        self.observations = np.concatenate([con.recv() for con in self.main_connects],axis=0)
        self.are_news = np.concatenate([con.recv() for con in self.main_connects],axis=0)

    def close(self):
        for proc in self.procs:
            proc.terminate()

    def process_start(conn, num_envs):
        envs = Envs(num_envs)
        while True:
            conn.send(envs.get_observations())
            conn.send(envs.are_new())
            actions = conn.recv()
            envs.set_actions(actions)
            conn.send(envs.get_rewards())

    def get_observations(self):
        return self.observations

    def are_new(self):
        return self.are_news

    def get_rewards(self):
        return self.rewards

    def observation_space(self):
        return self.example_env.observation_space.shape

    def action_space(self):
        space = self.example_env.action_space
        if isinstance(space,gym.spaces.Box):
            return space.shape
        else:
            return space.n

    def manipulate_actions(self,action_vecs):
        new_actions = []
        for i in range(self.num_envs):
            env = self.example_env
            action = process_action(env,action_vecs[i])

            prop_randomize_invalid = 0.3
            if not env.action_space.contains(action) and random.random() < prop_randomize_invalid:
                action = env.action_space.sample()

            prop_randomize_valid = 0.1
            if env.action_space.contains(action) and random.random() < prop_randomize_valid:
                action = env.action_space.sample()

            new_actions.append(action)

        return new_actions

    def set_actions(self,actions):
        for con,proc_actions in zip(self.main_connects,self.proc_splits.split(actions)):
            con.send(proc_actions)

        self.rewards = np.concatenate([con.recv() for con in self.main_connects],axis=0)
        self.observations = np.concatenate([con.recv() for con in self.main_connects],axis=0)
        self.are_news = np.concatenate([con.recv() for con in self.main_connects],axis=0)

    def random_actions(self):
        return [self.example_env.action_space.sample() for env in range(self.num_envs)]

if __name__ == "__main__":
    size = 128
    envs = MultiProcessEnvs(size)
    base_env = new_env()
    start = time.time()
    for i in range(1000):
        envs.set_actions([base_env.action_space.sample() for _ in range(size)])
        end = time.time()
        print(end-start)
        print((end-start)/(1+i))
