import numpy as np

a = [62, 255, 8, 255]
b = [0, 255, 0, 255]
c = [70, 255, 70, 255]

a = np.asarray(a)
b = np.asarray(b)
c = np.asarray(c)

print(a >= b)
print(a <= c)