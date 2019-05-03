import random
import tensorflow as tf

class TensorLossBallancer:
    '''
    Idea is that in a two loss neural network when both losses are important,
    one loss should not come to be too much bigger than the other, or else gradient
    descent will eventually forget that the smaller loss ever existed.

    This batch sequence based ballancer is designed to make the model loss magnitude independent.

    Like other batch sequence based algorithms, it assumes that batches are independently sampled.
    It does not assume anything about batches internally, they can be as correlated as you could want
    '''

    def __init__(self, A_prop,B_prop,ADAPT_COEF=0.95, MAG_ADJUST_COEF=0.99, DECAY_RATE=1.0, ESPILON=0.00001):
        self.A_prop = A_prop
        self.B_prop = B_prop
        self.ADAPT_COEF = ADAPT_COEF
        self.MAG_ADJUST_COEF = MAG_ADJUST_COEF
        self.DECAY_RATE = DECAY_RATE
        self.EPSION = ESPILON
        self.a_adj_state = None
        self.b_adj_state = None

    def adjust(self,a,b):
        # assumes losses are positive, which they should always be

        with tf.variable_scope("loss_ballancer"):
            if self.a_adj_state is None:
                self.a_adj_state = tf.Variable(tf.ones(1),dtype=tf.float32,trainable=False)
                self.b_adj_state = tf.Variable(tf.ones(1),dtype=tf.float32,trainable=False)

            a_adj = tf.stop_gradient(self.a_adj_state)
            b_adj = tf.stop_gradient(self.b_adj_state)
            new_a = a * (a_adj * self.A_prop)
            new_b = b * (b_adj * self.B_prop)
            a_adj_a = self.ADAPT_COEF * a_adj + (1-self.ADAPT_COEF) * ((b*b_adj) / (a+self.EPSION))
            b_adj_a = self.ADAPT_COEF * b_adj + (1-self.ADAPT_COEF) * ((a*a_adj) / (b+self.EPSION))
            a_adj_m = (a_adj_a**self.MAG_ADJUST_COEF) * ((1/b_adj_a)**(1-self.MAG_ADJUST_COEF))
            b_adj_m = (b_adj_a**self.MAG_ADJUST_COEF) * ((1/a_adj_a)**(1-self.MAG_ADJUST_COEF))
            stateful_ops = tf.group(tf.assign(self.a_adj_state,a_adj_m),
                     tf.assign(self.b_adj_state,b_adj_m))
        return new_a, new_b, stateful_ops


class LossBallancer:
    '''
    Idea is that in a two loss neural network when both losses are important,
    one loss should not come to be too much bigger than the other, or else gradient
    descent will eventually forget that the smaller loss ever existed.

    This batch sequence based ballancer is designed to make the model loss magnitude independent.

    Like other batch sequence based algorithms, it assumes that batches are independently sampled.
    It does not assume anything about batches internally, they can be as correlated as you could want
    '''

    def __init__(self, ADAPT_COEF, MAG_ADJUST_COEF, DECAY_RATE=0.9999, ESPILON=0.0001):
        self.ADAPT_COEF = ADAPT_COEF
        self.MAG_ADJUST_COEF = MAG_ADJUST_COEF
        self.DECAY_RATE = DECAY_RATE
        self.EPSION = ESPILON
        self.DECAY_RATE = DECAY_RATE
        self.a_adj = 1.0
        self.b_adj = 1.0

    def adjust(self,a,b):
        a_adj = self.a_adj
        b_adj = self.b_adj
        new_a = a * a_adj
        new_b = b * b_adj
        #print(a_adj,b_adj)
        self.a_adj = self.ADAPT_COEF * a_adj + (1-self.ADAPT_COEF) * ((b*b_adj) / (a+self.EPSION))
        self.b_adj = self.ADAPT_COEF * b_adj + (1-self.ADAPT_COEF) * ((a*a_adj) / (b+self.EPSION))
        a_adj = self.a_adj
        b_adj = self.b_adj
        self.a_adj = self.DECAY_RATE * (a_adj**self.MAG_ADJUST_COEF) * ((1/b_adj)**(1-self.MAG_ADJUST_COEF))
        self.b_adj = self.DECAY_RATE * (b_adj**self.MAG_ADJUST_COEF) * ((1/a_adj)**(1-self.MAG_ADJUST_COEF))
        return new_a, new_b

def loss_generator():
    #for x in range(1,80):
    #    yield x,1.0+random.random()
    #    yield x,0.0
    for y in range(1,40):
        yield (1.0+random.random()),0.1*(0.1+random.random())
        #yield (1.0+random.random()),0.1*(0.1+random.random())

def test_tensor_loss_ballancer():
    a_tensor = tf.placeholder(tf.float32,())
    b_tensor = tf.placeholder(tf.float32,())
    ballancer = TensorLossBallancer(1.1,0.1,0.95,0.99)
    newa,newb,stat_vals = ballancer.adjust(a_tensor,b_tensor)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for a,b in loss_generator():
            new_a, new_b, _ = sess.run((newa,newb,stat_vals),feed_dict={
                a_tensor: a,
                b_tensor: b,
            })
            print("{}\t{}".format(new_a,new_b))

def test_ballancer():
    ballancer = LossBallancer(0.95,0.99)
    for a,b in loss_generator():
        new_a,new_b = ballancer.adjust(a,b)
        print("{}\t{}".format(new_a,new_b))

if __name__ == "__main__":
    test_tensor_loss_ballancer()
    #test_ballancer()
