import random

class SampleStorage:
    '''
    Stores a fixed number of data on a FIFO output policy

    Also implements a sample procedure which unformly smaples the data
    '''
    def __init__(self,maxsize):
        self.data = []
        self.idx = 0
        self.maxsize = maxsize

    def full(self):
        return self.maxsize <= len(self.data)

    def sample(self):
        return random.choice(self.data)

    def put(self, element):
        if self.full():
            new_idx = (self.idx + 1) % (self.maxsize)
            old_idx = self.idx
            self.data[old_idx] = element
            self.idx = new_idx
        else:
            self.data.append(element)

def sample_storeage_test():
    ss = SampleStorage(100)
    for x in range(1000):
        #print(ss.data)
        ss.put(x)
        print(ss.sample())
sample_storeage_test()
