import pandas as pd
import joblib

model = joblib.load("model.pkl")
test_data = pd.read_csv("data/test_set.csv")

fraud_rows = test_data[test_data["Class"] == 1].drop("Class", axis=1)
probabilities = model.predict_proba(fraud_rows)[:, 1]  # confidence of "fraud" for each row

best_index = probabilities.argmax()
confident_fraud_row = fraud_rows.iloc[best_index].to_dict()

print("Confidence:", probabilities[best_index])
print("CONFIDENT_FRAUD_EXAMPLE =", confident_fraud_row)