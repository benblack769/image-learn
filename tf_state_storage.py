import tensorflow as tf
import numpy as np

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

class StorageAccessor:
    def __init__(self,keep_size,input_shape,action_shape):
        self.keep_size = keep_size
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.storage = StoredData(
            prev_input=Storage(keep_size,input_shape),
            cur_input=Storage(keep_size,input_shape),
            next_input=Storage(keep_size,input_shape),
            action=Storage(keep_size,action_shape),
            true_reward=Storage(keep_size,[1])
        )
        self.storage_list = self.storage.listify()
        self.current_size = tf.Variable(np.zeros(None,dtype=np.int32),dtype=tf.int32)
        #self.cached_data = tf.Variable(np.zeros(self.keep_size,dtype=np.float32),dtype=tf.float32)

    def sample_idxs(self,batch_size):
        maxval = tf.math.maximum(self.current_size,1)
        idxs = tf.random_uniform((batch_size,),dtype=tf.int32,minval=0,maxval=maxval)
        return idxs

    def replace_idxs(self,batch_size):
        return self.sample_idxs(batch_size)

    def access(self,idxs,batch_size):
        return StoredData.from_list([stor.get_idxs(idxs) for stor in self.storage_list])

    def add_data(self,batch_size,new_stored_data):
        # sequential add idxs for when self.current_size is low
        sequential_add_idxs = tf.range(batch_size,dtype=tf.int32) + self.current_size
        # random add idxs for when self.current_size is high
        rand_add_idxs = self.replace_idxs(batch_size)

        update_idxs = tf.where(
            condition=self.current_size < self.keep_size-batch_size*2,
            x=sequential_add_idxs,
            y=rand_add_idxs
        )

        new_size = tf.math.minimum(self.current_size+batch_size,self.keep_size)
        size_update = tf.assign(self.current_size,new_size)

        store_update = self.add_data_at(new_stored_data,update_idxs)
        update = tf.group([size_update,store_update])
        return update

    def add_data_at(self,new_stored_data,update_idxs):
        update = tf.group([stor.update_idxs(update_idxs,val) for stor,val in zip(self.storage_list,new_stored_data.listify())])
        return update

class WeightedAccessor(StorageAccessor):
    def __init__(self,keep_size,input_shape,action_shape):
        StorageAccessor.__init__(self,keep_size,input_shape,action_shape)
        self.weights = Storage(keep_size,[])
        #self.degrade_count = tf.Variable(np.zeros(None,dtype=np.int32),dtype=tf.int32)

    def replace_idxs(self,batch_size):
        IDXS_MULTIPLIER = 3
        new_size = batch_size * IDXS_MULTIPLIER
        base_idxs = self.sample_idxs(new_size)
        idx_weights = self.weights.get_idxs(base_idxs)

        #throw out low value weights
        bad_values, bad_idxs = tf.math.top_k(-idx_weights,k=batch_size)

        actual_bad_idxs = tf.gather(base_idxs,bad_idxs,axis=0)

        return actual_bad_idxs

    def update_weights_at(self,weights,idxs):
        return self.weights.update_idxs(idxs,weights)

    def add_data_at(self,new_stored_data,update_idxs):
        large_val = tf.cast((tf.ones_like(update_idxs)*10000),tf.float32)
        store_update = StorageAccessor.add_data_at(self,new_stored_data,update_idxs)
        weight_update = self.weights.update_idxs(update_idxs,large_val)
        update = tf.group([store_update,weight_update])
        return update

class Storage:
    def __init__(self,keep_size,shape):
        self.keep_size = keep_size
        self.data = tf.Variable(np.zeros([self.keep_size]+shape,dtype=np.float32),dtype=tf.float32)

    def get_idxs(self,idxs):
        return tf.gather(self.data,idxs,axis=0)

    def update_idxs(self,idxs,new_vals):
        return tf.scatter_update(self.data,idxs,new_vals)


def data_generator(state_storage,batch_size):
    idxs = state_storage.sample_idxs(batch_size)
    new_datas = state_storage.access(idxs,batch_size)
    return new_datas,idxs
