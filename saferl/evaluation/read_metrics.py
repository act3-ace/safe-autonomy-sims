import pickle

file_path = "/tmp/data/inspection/gen-metrics.pkl"

with open(file_path, 'rb') as file:
    data = pickle.load(file)

print("data")
print(data)
print()

print('blue0 states')
print(data.participants['blue0'].events[0].metrics["TrueState"])
print()
