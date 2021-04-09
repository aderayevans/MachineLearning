import pandas as pd

data = pd.read_csv('baitap1.csv', delimiter=',')

print(data)
print('/.../')
print(data.iloc[:,2:3])
print('/.../')
print(data.iloc[4:10,:])
print('/.../')
print(data.iloc[4:5,0:2])
print('/.../')
x = data.iloc[:,1:2]
y = data.iloc[:,2:3]
print(x)
print('/.../')
print(y)
import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.title('Bai tap 1')
plt.xlabel('Tuoi')
plt.ylabel('Can nang')
plt.xticks([w*7*24 for w in range(10)], ['%i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()

for index in range (1, 51):
	if (index % 2 != 0): print(index)


