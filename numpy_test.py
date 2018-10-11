import numpy as np

arg = np.arange(100).reshape((10,10))
print(arg)

print(arg[[(1,2),(3,4)]])

res = np.digitize(0.4,[0.2,0.3])
print(res)
print(np.add.accumulate([0.1,0.2,0.3]))
