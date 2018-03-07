import datetime
import time
import numpy as np

start = time.time()

for i in range(10000):

    x = np.random.rand(100,100)
    y = np.sort(x)

duration = 12050




print(duration)
print(datetime.timedelta(seconds=duration))
print("{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds)))



