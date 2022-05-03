import numpy as np

a = np.random.randint(0,10,(50,60,3))
# print(a)
print('***')
b = a.reshape(-1,3)
# print(b)
print('***')
c = b.reshape(50,60,3)
# print(c)
print('***')
d = np.random.randint(0,2,(50,60))
print(d)
print('***')
e = a*d
print(e)