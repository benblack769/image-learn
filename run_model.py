import gym
from gym_wrapper import Envs
import tensorflow as tf
from model import Runner
import numpy as np

def new_randvec(RAND_SIZE):
    return np.random.binomial(size=[NUM_ENVS,RAND_SIZE], n=1, p=0.5).astype(np.float32)

envs = Envs(1)
env = envs.envs[0]
#env = gym.wrappers.Monitor(env, "trained_recording")
env.reset()

SAVE_NAME = "save_model/model.ckpt"

action_shape = list(envs.action_space())
observation_shape = list(envs.observation_space())

NUM_ENVS = 1

LAYER_SIZE = 48
RAND_SIZE = 8
model = Runner(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, SAVE_NAME)

    prev_input = np.zeros([NUM_ENVS]+observation_shape,dtype=np.float32)

    RAND_SIZE = 8

    steps = 0
    done = False
    while True:
        current_input = envs.get_observations()
        #actions = [env.action_space.sample()]
        if steps % 4 == 0:
            randvec = new_randvec(RAND_SIZE)
        actions = model.calc_action_data(sess,current_input,prev_input,randvec)
        env.render()

        envs.set_actions(actions)
        print(actions)
        #print(env.action_space.n)
        done = envs.are_new()[0]

        prev_input = current_input
        steps += 1
        #env.render()
        #print(action)
        #print(reward)
        #print(done)
        #print(info)
    print(steps)
    env.close()
