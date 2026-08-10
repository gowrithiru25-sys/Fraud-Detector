import pandas as pd

data = pd.read_csv("data/creditcard.csv")

fraud_row = data[data["Class"] == 1].iloc[0].drop("Class").to_dict()
legit_row = data[data["Class"] == 0].iloc[0].drop("Class").to_dict()

print("FRAUD_EXAMPLE =", fraud_row)
print()
print("LEGIT_EXAMPLE =", legit_row)