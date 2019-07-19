import gym
from gym_wrapper import Envs
import tensorflow as tf
import numpy as np
from model import MainModel

def new_randvec(RAND_SIZE):
    return np.random.binomial(size=[NUM_ENVS,RAND_SIZE], n=1, p=0.5).astype(np.float32)


class Runner:
    def __init__(self,observation_shape,action_shape,LAYER_SIZE,RAND_SIZE):
        NUM_ENVS = 1
        self.model = MainModel(action_shape,observation_shape,LAYER_SIZE,RAND_SIZE)

        self.runner_true_input1 = tf.placeholder(shape=[NUM_ENVS,]+observation_shape,dtype=tf.float32)
        self.runner_true_input2 = tf.placeholder(shape=[NUM_ENVS,]+observation_shape,dtype=tf.float32)
        self.runner_current_randvec = tf.placeholder(shape=[NUM_ENVS,RAND_SIZE],dtype=tf.float32)

        #self.runner_gen_action,_ = self.model.calc_sample_batch(self.runner_true_input1,self.runner_true_input2,8,NUM_ENVS)
        self.runner_gen_action = self.model.calc_action(self.runner_true_input1,self.runner_true_input2,self.runner_current_randvec)

    def calc_action(self,sess,prev_input,input,randvec):
        actions = sess.run(self.runner_gen_action,feed_dict={
            self.runner_true_input1:prev_input,
            self.runner_true_input2:input,
            self.runner_current_randvec:randvec,
        })
        return actions


envs = Envs(1)
env = envs.envs[0]
#env = gym.wrappers.Monitor(env, "trained_recording")
env.reset()

SAVE_NAME = "save_model/model.ckpt"

action_shape = list(envs.action_space())
observation_shape = list(envs.observation_space())

NUM_ENVS = 1

LAYER_SIZE = 128
RAND_SIZE = 8
model = Runner(observation_shape,action_shape,LAYER_SIZE,RAND_SIZE)

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
        actions = model.calc_action(sess,current_input,prev_input,randvec)
        actions = [actions[0]]
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
