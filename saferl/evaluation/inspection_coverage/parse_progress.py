import pandas as pd
import matplotlib.pyplot as plt
df  = pd.read_csv("progress.csv")
df.plot()  # plots all columns against index
df.plot(kind='scatter',x='training_iteration',y='episode_reward_mean') # scatter plot
 
plt.savefig("output.png")
