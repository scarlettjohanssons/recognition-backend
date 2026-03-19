import pandas as pd

df_main = pd.read_csv("app/data/big_labels.csv")
df_mixed = pd.read_csv("app/data/mixed_labels.csv")

df_full = pd.concat([df_main, df_mixed], ignore_index=True)
df_full.to_csv("app/data/extended_labels.csv", index=False)