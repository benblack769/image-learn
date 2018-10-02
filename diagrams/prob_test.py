import numpy as np
import time

TOT_SIZE = 1000
SAMP_SIZE = 50

def norm_pos(x):
    return x / np.sum(x)

times_chosen = 1+np.zeros(TOT_SIZE)
times_max = 1+np.zeros(TOT_SIZE)

for x in range(100000):
    prob_max = times_max/(times_chosen+0.0001)
    cur_probs = norm_pos((0.001+prob_max * np.random.uniform(TOT_SIZE)))
    sample = np.random.choice(np.arange(TOT_SIZE),size=SAMP_SIZE,p=cur_probs)
    #print(sample)
    max_val = max(sample)
    times_chosen[sample] += 1
    times_max[max_val] += 1

print(prob_max)
