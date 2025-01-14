import pandas as pd

df = pd.read_csv("data/tic-tac-toe.csv")

print(df["target"] == True)
for i in range(len(df["target"])):
    if not df["target"][i]:
        print(i)
        break
