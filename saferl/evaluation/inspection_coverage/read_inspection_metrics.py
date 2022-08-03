import pickle
import matplotlib.pyplot as plt

file_path = "/tmp/data/inspection/gen-metrics.pkl"

with open(file_path, 'rb') as file:
    data = pickle.load(file)

print("data")
print(data)
print()

print('blue0 states')
print(data.participants['blue0'].events[0].metrics["TrueState"])
print(data.participants['blue0'].events[0].metrics["TotalReward"])
# print(data.participants['blue0'].events[0].metrics["Dones"])

# print(type(data.participants['blue0'].events[0].metrics["TrueState"].values()[0]))
# print(type(data.participants['blue0'].events[0].metrics["TrueState"].keys()[0]))

print()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# ax = fig.add_subplot(111)

ax.scatter(0, 0, 0, c=0, s=10)
# ax.scatter(0,0, c=0, s = 10)


for agent in data.participants['blue0'].events[0].metrics["TrueState"].keys():
    for item in data.participants['blue0'].events[0].metrics["TrueState"][agent]:
        state_dict = data.participants['blue0'].events[0].metrics["TrueState"][agent][item]
        pos = state_dict['position']
        velocity = state_dict['velocity']
        ax.scatter(pos[0], pos[1], pos[2], c=4, s=1)
        # ax.scatter(pos[0], pos[1],c=5, s = 1)

plt.show()
plt.savefig('3dresults.png')

