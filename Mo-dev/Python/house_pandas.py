import matplotlib.pyplot as plt
import pandas as pd

print(pd.__version__)
dataframe = pd.read_csv("data/house/house.csv")
print(dataframe.describe())

dataframe.hist(bins=20)
plt.show()

plt.scatter(dataframe["surface"], dataframe["loyer"])
plt.show()